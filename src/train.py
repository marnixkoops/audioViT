import logging
import multiprocessing
import os
import warnings
from datetime import datetime
from pprint import pformat

import audiomentations as A
import librosa
import lightning as L
import numpy as np
import pandas as pd
import seaborn as sns
import timm
import torch
import torchaudio
import torchmetrics
from joblib import Parallel, delayed
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm.notebook import tqdm

sns.set(
    rc={
        "figure.figsize": (8, 4),
        "figure.dpi": 240,
    }
)
sns.set_style("darkgrid", {"axes.grid": False})
sns.set_context("paper", font_scale=0.6)

torch.set_float32_matmul_precision("high")
warnings.simplefilter("ignore")


class cfg:
    experiment_name = "baseline"
    data_path = "../data"

    debug_run = True  # run on 250 rows for debugging
    experiment_run = False  # run on a stratified 80% data sample for experimentation
    competition_run = False  # run on all data

    normalize_waveform = True
    sample_rate = 32000
    n_fft = 2048
    hop_length = 512
    window_length = None
    melspec_hres = 128
    melspec_wres = 312
    freq_min = 20
    freq_max = 16000
    log_scale_power = 2
    max_decibels = 80
    frame_duration = 5
    frame_rate = sample_rate / hop_length

    vit_b0 = "efficientvit_b0.r224_in1k"
    # vit_b1 = "efficientvit_b1.r224_in1k"
    # vit_b1 = "efficientvit_b1.r288_in1"
    # effnet_b0 = "tf_efficientnetv2_b0.in1k"
    # effnet_b1 = "tf_efficientnetv2_b1.in1k"
    # vit_m0 = "efficientvit_m0.r224_in1k"
    # vit_m1 = "efficientvit_m1.r224_in1k"
    backbone = vit_b0

    num_classes = 182
    add_secondary_labels = True

    accelerator = "gpu"
    precision = "16-mixed"
    n_workers = multiprocessing.cpu_count() - 4

    n_epochs = 25
    batch_size = 64
    val_ratio = 0.25

    lr_min = 1e-7
    lr_max = 1e-4
    weight_decay = 1e-6
    label_smoothing = 0.0
    fused_adamw = True

    timestamp = datetime.now().replace(microsecond=0)
    run_tag = f"{timestamp}_{backbone}_{experiment_name}_val_{val_ratio}_lr_{lr_max}_decay_{weight_decay}"

    if debug_run:
        run_tag = f"{timestamp}_{backbone}_debug"
        accelerator = "cpu"
        n_epochs = 10
        batch_size = 16
        fused_adamw = False
        num_classes = 10


def define_logger():
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(f"../logs/{cfg.run_tag}.log"),
    ]

    if cfg.debug_run:
        handlers = [logging.StreamHandler()]

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format=" %(asctime)s [%(threadName)s] 🐦‍🔥 %(message)s",
        handlers=handlers,
        force=True,  # reconfigure root logger, in case of rerunning -> ensures new file
    )

    return logger


def get_config(cfg) -> None:
    cfg_dictionary = {
        key: value
        for key, value in cfg.__dict__.items()
        if not key.startswith("__") and not callable(key)
    }
    logger.info(f"{'—' * 80}")
    logger.info(f"Config: \n{pformat(cfg_dictionary, indent=1)}")
    return cfg_dictionary


def load_metadata(data_path: str) -> pd.DataFrame:
    logger.info(f"Loading prepared dataframes from {data_path}")
    model_input_df = pd.read_csv(f"{data_path}/model_input_df.csv")
    sample_submission = pd.read_csv(f"{data_path}/sample_submission.csv")

    if cfg.debug_run:
        logger.info("Running debug: sampling data to 10 species and 250 samples")
        top_10_labels = model_input_df["primary_label"].value_counts()[0:10].index
        model_input_df = model_input_df[
            model_input_df["primary_label"].isin(top_10_labels)
        ]
        model_input_df = model_input_df.sample(100).reset_index(drop=True)

    if cfg.experiment_run:
        logger.info("Running experiment: stratified sampling 80% of data")
        model_input_df, _ = train_test_split(
            model_input_df,
            test_size=0.20,
            stratify=model_input_df["primary_label"],
            shuffle=True,
            random_state=None,
        )

    logger.info(f"Dataframe shape: {model_input_df.shape}")

    return model_input_df, sample_submission


def read_waveform(filename: str) -> np.ndarray:
    filepath = f"{cfg.data_path}/train_windows_n5_s5_c9/{filename}"
    waveform, _ = librosa.load(filepath, sr=cfg.sample_rate)
    return waveform


def read_waveforms_parallel(model_input_df: pd.DataFrame):
    logger.info("Parallel Loading waveforms")
    waveforms = Parallel(n_jobs=cfg.n_workers)(
        delayed(read_waveform)(filename)
        for filename in tqdm(model_input_df["window_filename"], desc="Loading waves")
    )
    logger.info("Finished loadeding waveforms")
    return waveforms


def create_label_map(submission_df: pd.DataFrame) -> dict:
    logging.info("Creating label mappings")
    cfg.labels = submission_df.columns[1:]
    cfg.num_classes = len(cfg.labels)

    class_to_label_map = dict(zip(cfg.labels, np.arange(cfg.num_classes)))

    return class_to_label_map


def pad_or_crop_waveforms(waveforms: list) -> list:
    logging.info("Padding or cropping waveforms to desired duration")
    desired_length = cfg.sample_rate * cfg.frame_duration

    def _pad_or_crop(waveform: np.ndarray) -> np.ndarray:
        length = len(waveform)
        if length < desired_length:  # repeat if waveform too small
            repeat_length = desired_length - length
            waveform = np.concatenate([waveform, waveform[:repeat_length]])
            length = len(waveform)

            if length < desired_length:  # repeat if waveform still too small
                repeat_length = desired_length - length
                waveform = np.concatenate([waveform, waveform[:repeat_length]])
                length = len(waveform)

        if length > desired_length:  # crop if waveform is too big
            offset = np.random.randint(0, length - desired_length)
            waveform = waveform[offset : offset + desired_length]

        return waveform

    waveforms = [_pad_or_crop(wave) for wave in tqdm(waveforms, desc="Padding waves")]

    return waveforms


def define_train_augmentations(augment: bool = cfg.augment) -> A.Compose:
    if augment:
        train_augmentations = A.Compose(
            [
                A.LowPassFilter(
                    min_cutoff_freq=150.0,
                    max_cutoff_freq=7500.0,
                    min_rolloff=12.0,
                    max_rolloff=24.0,
                    zero_phase=False,
                    p=0.5,
                ),
                A.Gain(min_gain_db=-12, max_gain_db=12, p=0.5),
                A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.05, p=0.5),
                # A.SevenBandParametricEQ(min_gain_db=-12.0, max_gain_db=12.0, p=0.5),
                # A.AirAbsorption(
                #     min_distance=10.0,
                #     max_distance=100.0,
                #     p=0.25,
                # ),
                # A.PitchShift(min_semitones=-5.0, max_semitones=5.0, p=0.5),
                # A.BandPassFilter(min_center_freq=100.0, max_center_freq=6000, p=1.0),
                # A.TimeMask(
                #     min_band_part=0.1,
                #     max_band_part=0.15,
                #     fade=True,
                #     p=0.5,
                # ),
                # A.BitCrush(min_bit_depth=5, max_bit_depth=14, p=0.25)
            ]
        )
    else:
        train_augmentations = None

    logger.info(f"Augmentations: \n{pformat(train_augmentations.transforms)}")

    return train_augmentations


class BirdDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        waveforms: list,
        add_secondary_labels: bool = cfg.add_secondary_labels,
        augmentations: list = None,
    ):

        self.df = df
        self.waveforms = waveforms
        self.num_classes = cfg.num_classes
        self.class_to_label_map = class_to_label_map
        self.add_secondary_labels = add_secondary_labels

        self.augmentations = augmentations

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_mels=cfg.melspec_hres,
            f_min=cfg.freq_min,
            f_max=cfg.freq_max,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            normalized=cfg.normalize_waveform,
            center=True,
            pad_mode="reflect",
            norm="slaney",
            mel_scale="slaney",
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=cfg.max_decibels
        )

    def create_target(
        self,
        primary_label: str,
        secondary_labels: list,
        secondary_label_weight: float = 1,
    ) -> torch.tensor:
        target = torch.zeros(self.num_classes, dtype=torch.int64)
        # primary_target = torch.tensor([0], dtype=torch.int64)

        if primary_label != "nocall":
            primary_label = self.class_to_label_map[primary_label]
            target[primary_label] = 1
            primary_target = torch.tensor(primary_label, dtype=torch.int64)

            if self.add_secondary_labels:
                secondary_labels = eval(secondary_labels)
                for label in secondary_labels:
                    if label != "" and label in self.class_to_label_map.keys():
                        target[self.class_to_label_map[label]] = secondary_label_weight

        return target, primary_target

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        waveform = self.waveforms[idx]
        primary_label = self.df.iloc[idx]["primary_label"]
        secondary_labels = self.df.iloc[idx]["secondary_labels"]
        augmentations = self.augmentations

        if augmentations is not None:
            waveform = augmentations(waveform, sample_rate=cfg.sample_rate)

        waveform = torch.tensor(waveform, dtype=torch.float32).squeeze()

        melspec = self.db_transform(self.mel_transform(waveform)).type(torch.uint8)
        melspec = melspec - melspec.min()
        melspec = melspec / melspec.max() * 255

        target, primary_target = self.create_target(
            primary_label=primary_label, secondary_labels=secondary_labels
        )

        return melspec, target, primary_target


if __name__ == "__main__":
    logger = define_logger()
    config_dictionary = get_config(cfg)

    model_input_df, sample_submission = load_metadata(data_path=cfg.data_path)
    class_to_label_map = create_label_map(submission_df=sample_submission)

    waveforms = read_waveforms_parallel(model_input_df=model_input_df)
    waveforms = pad_or_crop_waveforms(waveforms=waveforms)

    # grouped split on sample index to keep different windows from the same sample
    # together if splitting randomly this can be considered as a form of leakage
    # validating on a windowed waveform while windows of the same waveform were used for
    # training is easier than classifying a waveform from a different sample, which is
    # the case in practice

    logger.info(f"Splitting {len(waveforms)} waveforms into train/val: {cfg.val_ratio}")
    n_splits = int(round(1 / cfg.val_ratio))
    kfold = StratifiedGroupKFold(n_splits=n_splits, shuffle=True)

    for fold_index, (train_index, val_index) in enumerate(
        kfold.split(
            X=model_input_df,
            y=model_input_df["primary_label"],
            groups=model_input_df["sample_index"],
        )
    ):
        break  # run a single split for now

    train_df = model_input_df.iloc[train_index]
    val_df = model_input_df.iloc[val_index]

    train_waveforms = [waveforms[i] for i in train_index]
    val_waveforms = [waveforms[i] for i in val_index]

    train_augmentations = define_train_augmentations()
    train_dataset = BirdDataset(
        df=train_df, waveforms=train_waveforms, augmentations=train_augmentations
    )
    val_dataset = BirdDataset(df=val_df, waveforms=val_waveforms)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        drop_last=True,
        num_workers=cfg.n_workers,
        persistent_workers=True,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=cfg.n_workers,
        persistent_workers=True,
        pin_memory=True,
    )

    class EfficientViT(L.LightningModule):
        def __init__(self):
            super().__init__()

            self.vit = timm.create_model(
                cfg.backbone,
                pretrained=True,
                num_classes=cfg.num_classes,
            )

            self.imagenet_normalize = transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            )

            self.loss_function = nn.CrossEntropyLoss(
                label_smoothing=cfg.label_smoothing
            )
            self.accuracy = torchmetrics.Accuracy(
                task="multiclass", num_classes=cfg.num_classes
            )
            self.f1_macro = torchmetrics.F1Score(
                task="multilabel",
                num_labels=cfg.num_classes,
                average="macro",
                ignore_index=0,
            )
            self.f1_weighted = torchmetrics.F1Score(
                task="multilabel",
                num_labels=cfg.num_classes,
                average="weighted",
                ignore_index=0,
            )
            self.auroc = torchmetrics.AUROC(
                task="multilabel",
                num_labels=cfg.num_classes,
                average="macro",
            )

        def forward(self, x):
            x = x.unsqueeze(1).expand(-1, 3, -1, -1)  # go from HxW → 3xHxW
            x = x.float() / 255
            x = self.imagenet_normalize(x)
            out = self.vit(x)

            return out

        def training_step(self, batch, batch_idx):
            x, y, y_primary = batch
            y_pred = self(x)

            loss = self.loss_function(y_pred, y_primary)
            train_acc = self.accuracy(y_pred.softmax(dim=1), y_primary)
            train_f1_macro = self.f1_macro(y_pred.softmax(dim=1), y)
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
                "train_acc_1",
                train_acc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "train_f1_macro_ignore",
                train_f1_macro,
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
            x_val, y_val, y_primary_val = batch
            y_pred = self(x_val)

            val_loss = self.loss_function(y_pred, y_primary_val)
            val_acc = self.accuracy(y_pred.softmax(dim=1), y_primary_val)
            val_f1_weighted = self.f1_weighted(y_pred.softmax(dim=1), y_val)
            val_f1_macro = self.f1_macro(y_pred.softmax(dim=1), y_val)
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
                "val_acc_1",
                val_acc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "val_f1_weighted_ignore",
                val_f1_weighted,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "val_f1_macro_ignore",
                val_f1_macro,
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

        def on_train_epoch_end(self):
            metrics = self.trainer.progress_bar_callback.get_metrics(trainer, model)
            metrics.pop("v_num", None)
            metrics.pop("train_loss_step", None)
            for key, value in metrics.items():
                metrics[key] = round(value, 5)
            logger.info(f"Epoch {self.trainer.current_epoch}: {metrics}")

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(
                params=self.parameters(),
                lr=cfg.lr_max,
                weight_decay=cfg.weight_decay,
                fused=cfg.fused_adamw,
            )
            lr_scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=cfg.n_epochs, T_mult=1, eta_min=cfg.lr_min, last_epoch=-1
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

    csv_logger = None
    if not cfg.debug_run:
        os.environ["PJRT_DEVICE"] = "GPU"  # fix for G Cloud to avoid XLA/autocast clash
        csv_logger = L.pytorch.loggers.CSVLogger(save_dir="../logs/")
        csv_logger.log_hyperparams(config_dictionary)

    progress_bar = TQDMProgressBar(process_position=1)
    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=cfg.patience,
        verbose=True,
        mode="min",
    )

    model = EfficientViT()
    trainer = L.Trainer(
        fast_dev_run=False,
        enable_model_summary=True,
        max_epochs=cfg.n_epochs,
        accelerator=cfg.accelerator,
        precision=cfg.precision,
        callbacks=[progress_bar, early_stopping],
        logger=csv_logger,
        log_every_n_steps=10,
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=None,
    )

    logger.info("Finished training")
    if not cfg.debug_run:
        logger.info("Saving model")
        filename = f"{cfg.run_tag}_epochs_{trainer.current_epoch}.ckpt"
        trainer.save_checkpoint(f"../model_objects/{filename}")
        logger.info(f"Saved model to filename: {filename}")
    logger.info("All done")
