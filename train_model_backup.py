import gc
import logging
import pickle
from datetime import datetime

import albumentations
import librosa
import lightning as L
import numpy as np
import pandas as pd
import seaborn as sns
import timm
import torch
import torchmetrics
import torchvision
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm.notebook import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format=" %(asctime)s [%(threadName)s] [%(levelname)s] 🐦‍🔥  %(message)s",
)

sns.set(
    rc={
        "figure.figsize": (8, 4),
        "figure.dpi": 240,
    }
)
sns.set_style("darkgrid", {"axes.grid": False})
sns.set_context("paper", font_scale=0.6)


class cfg:
    dry_run = True
    data_path = "../data"

    normalize_waveform = True
    sample_rate = 32000
    n_fft = 2048
    hop_length = 512
    window_length = None
    melspec_hres = 128
    melspec_wres = 312
    freq_min = 20
    freq_max = 16000
    log_scale_power = 3
    frame_duration = None
    max_decibels = 80
    frame_rate = sample_rate / hop_length

    vit_b0 = "efficientvit_b0.r224_in1k"
    # vit_b1 = "efficientvit_b1.r224_in1k"
    # vit_b1 = "efficientvit_b1.r288_in1"
    # effnet_b0 = "tf_efficientnetv2_b0.in1k"
    # effnet_b1 = "tf_efficientnetv2_b1.in1k"
    # vit_m0 = "efficientvit_m0.r224_in1k"
    # vit_m1 = "efficientvit_m1.r224_in1k"
    backbone = vit_b0

    n_epochs = 2
    lr_max = 1e-3
    weight_decay = 1e-6

    accelerator = "cpu"
    precision = "16-mixed"
    batch_size = 4
    n_workers = 6

    val_ratio = 0.25
    n_classes = 182

    timestamp = datetime.now().replace(microsecond=0)
    run_tag = f"{timestamp}_{backbone}_val_{val_ratio}_lr_{lr_max}_decay_{weight_decay}"


def load_metadata(data_path: str, dry_run: bool = cfg.dry_run):
    print(f"Loading prepared dataframes from {data_path}")
    model_input_df = pd.read_csv(f"{data_path}/model_input_df.csv")
    with open(f"{data_path}/labels.pkl", "rb") as file:
        labels = pickle.load(file)

    if dry_run:
        print("Dry running: sampling data to 10 species and 100 samples")
        top_10_bird_species = model_input_df["primary_label"].value_counts()[0:10].index
        model_input_df = model_input_df[
            model_input_df["primary_label"].isin(top_10_bird_species)
        ]
        model_input_df = model_input_df.sample(100)
        sampled_index = model_input_df.index.values
        labels = [labels[i] for i in sampled_index]

    print(f"Dataframe shape: {model_input_df.shape}")
    print(f"Labels shape: {len(labels)}")

    sample_submission = pd.read_csv(f"{data_path}/sample_submission.csv")
    cfg.labels = sample_submission.columns[1:]
    cfg.n_classes = len(cfg.labels)

    return model_input_df, labels, sample_submission


def load_waveform_train_windows(
    model_input_df: pd.DataFrame, normalize: bool = False
) -> list:
    print(f"Loading {len(model_input_df)} waveform train windows from {cfg.data_path}")

    def _load_single_waveform(filename: str) -> np.ndarray:
        filepath = f"{cfg.data_path}/train_windows_n5_s5_c9/{filename}"
        waveform, _ = librosa.load(
            filepath, sr=cfg.sample_rate, duration=cfg.frame_duration
        )
        if normalize:
            waveform = librosa.util.normalize(waveform)

        return waveform

    waveforms = [
        _load_single_waveform(filename)
        for filename in tqdm(model_input_df["window_filename"], desc="Loading waves")
    ]

    return waveforms


def create_melspec(waveform: np.ndarray) -> np.ndarray:
    melspec = librosa.feature.melspectrogram(
        y=waveform,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.melspec_hres,
        fmin=cfg.freq_min,
        fmax=cfg.freq_max,
        power=cfg.log_scale_power,
    )

    melspec = librosa.power_to_db(melspec, ref=cfg.max_decibels)
    melspec = melspec - melspec.min()
    melspec = (melspec / melspec.max() * 255).astype(np.uint8)

    return melspec


def _pad_or_crop_melspec(melspec: np.ndarray) -> np.ndarray:
    _, width = melspec.shape

    if width < cfg.melspec_wres:  # repeat if birdy melspec too small
        repeat_length = cfg.melspec_wres - width
        melspec = np.concatenate([melspec, melspec[:, :repeat_length]], axis=1)

    elif width > cfg.melspec_wres:  # crop if birdy melspec is too big
        offset = np.random.randint(0, width - cfg.melspec_wres)
        melspec = melspec[:, offset : offset + cfg.melspec_wres]

    return melspec


class BirdDataset(Dataset):
    def __init__(self, waveforms: list, labels: list):
        self.waveforms = waveforms
        self.labels = torch.tensor(labels, dtype=torch.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        waveform = self.waveforms[idx]
        label = self.labels[idx]

        melspec = create_melspec(waveform)
        melspec = _pad_or_crop_melspec(melspec)

        melspec = torch.tensor(melspec, dtype=torch.uint8)

        return melspec, label


if __name__ == "__main__":
    model_input_df, labels, sample_submission = load_metadata(
        data_path=cfg.data_path, dry_run=cfg.dry_run
    )
    waveforms = load_waveform_train_windows(model_input_df, normalize=False)

    # X_train, X_val, y_train, y_val = train_test_split(
    #     waveforms,
    #     labels,
    #     test_size=cfg.val_ratio,
    #     stratify=labels,
    #     shuffle=True,
    #     random_state=None,
    # )

    train_dataset = BirdDataset(X_train, y_train)
    val_dataset = BirdDataset(X_val, y_val)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        drop_last=True,
        num_workers=cfg.n_workers,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=cfg.n_workers,
        pin_memory=True,
    )

    class EfficientViT(L.LightningModule):
        def __init__(self):
            super().__init__()

            self.vit = timm.create_model(
                cfg.backbone,
                pretrained=True,
                num_classes=cfg.n_classes,
            )

            self.imagenet_normalize = transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            )

            self.loss_criterion = nn.CrossEntropyLoss()
            self.acc = torchmetrics.Accuracy(
                task="multiclass", num_classes=cfg.n_classes
            )
            self.auroc = torchmetrics.AUROC(
                task="multiclass", num_classes=cfg.n_classes
            )

        def forward(self, x):
            x = x.unsqueeze(1).expand(-1, 3, -1, -1)  # go from HxW → 3xHxW
            x = x.float() / 255
            x = self.imagenet_normalize(x)
            out = self.vit(x)

            return out

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_pred = self(x)
            loss = self.loss_criterion(y_pred, y)

            train_acc = self.acc(y_pred.softmax(dim=1), y)
            train_auroc = self.auroc(y_pred.softmax(dim=1), y)

            self.log(
                "train_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "train_acc",
                train_acc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "train_auroc",
                train_auroc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            return loss

        def validation_step(self, batch, batch_idx):
            x_val, y_val = batch
            y_pred = self(x_val)

            val_loss = self.loss_criterion(y_pred, y_val)
            val_acc = self.acc(y_pred.softmax(dim=1), y_val)
            val_auroc = self.auroc(y_pred.softmax(dim=1), y_val)

            self.log(
                "val_loss",
                val_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "val_acc",
                val_acc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "val_auroc",
                val_auroc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            return val_loss

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(
                params=self.parameters(),
                lr=cfg.lr_max,
                weight_decay=cfg.weight_decay,
                fused=False,
            )
            lr_scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=cfg.n_epochs, T_mult=1, eta_min=1e-6, last_epoch=-1
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "epoch",
                    "monitor": "val_loss",
                    "frequency": 1,
                },
            }

    # in case of run on Google instance, otherwise XLA/autocast argue and all is fucked
    # import os
    # os.environ["PJRT_DEVICE"] = "GPU"

    model = EfficientViT()
    trainer = L.Trainer(
        fast_dev_run=False,
        max_epochs=cfg.n_epochs,
        accelerator=cfg.accelerator,
        enable_model_summary=True,
        callbacks=[TQDMProgressBar(refresh_rate=1)],
        precision=cfg.precision,
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=None,
    )

    print("Finished training, saving model")
    cfg.timestamp = datetime.now().replace(microsecond=0)
    # trainer.save_checkpoint(f"model_objects/full_{cfg.run_tag}.ckpt")
    print("All done")
