import gc
import logging
import multiprocessing
import os
import pickle
import warnings
from datetime import datetime
from pprint import pformat

import albumentations
import librosa
import lightning as L
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile
import timm
import torch
import torchaudio
import torchmetrics
from joblib import Parallel, delayed
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm.notebook import tqdm

warnings.filterwarnings("ignore")
sns.set(
    rc={
        "figure.figsize": (8, 4),
        "figure.dpi": 240,
    }
)
sns.set_style("darkgrid", {"axes.grid": False})
sns.set_context("paper", font_scale=0.6)

# in case of run on Google instance, otherwise XLA/autocast argue and all is fucked
os.environ["PJRT_DEVICE"] = "GPU"


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
    log_scale_power = 2
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

    n_epochs = 3
    lr_max = 1e-3
    weight_decay = 1e-6

    accelerator = "cpu"
    precision = "16-mixed"
    batch_size = 8
    n_workers = 6

    val_ratio = 0.25
    n_classes = 182

    timestamp = datetime.now().replace(microsecond=0)
    run_tag = f"{timestamp}_{backbone}_val_{val_ratio}_lr_{lr_max}_decay_{weight_decay}"


def define_logger():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format=" %(asctime)s [%(threadName)s] [%(levelname)s] 🐦‍🔥  %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"../logs/{cfg.run_tag}.log"),
        ],
    )
    return logger


def get_config(cfg) -> None:
    cfg_dictionary = {
        key: value
        for key, value in cfg.__dict__.items()
        if not key.startswith("__") and not callable(key)
    }
    logger.info(f"{'—' * 80}")
    logger.info(f"\n {pformat(cfg_dictionary)}")
    return cfg_dictionary


def load_metadata(data_path: str, dry_run: bool = cfg.dry_run):
    logger.info(f"Loading prepared dataframes from {data_path}")
    model_input_df = pd.read_csv(f"{data_path}/model_input_df.csv")

    if dry_run:
        logger.info("Dry running: sampling data to 10 species and 100 samples")
        top_10_bird_species = model_input_df["primary_label"].value_counts()[0:10].index
        model_input_df = model_input_df[
            model_input_df["primary_label"].isin(top_10_bird_species)
        ]
        model_input_df = model_input_df.sample(100)
        sampled_index = model_input_df.index.values
        model_input_df = model_input_df.reset_index(drop=True)

    logger.info(f"Dataframe shape: {model_input_df.shape}")

    sample_submission = pd.read_csv(f"{data_path}/sample_submission.csv")

    return model_input_df, sample_submission


def create_label_map(submission_df: pd.DataFrame) -> dict:
    logging.info("Creating label mappings")
    cfg.labels = submission_df.columns[1:]
    cfg.n_classes = len(cfg.labels)

    class_to_label_map = dict(zip(cfg.labels, np.arange(cfg.n_classes)))

    return class_to_label_map


def load_waveform_train_windows(
    model_input_df: pd.DataFrame, normalize: bool = False
) -> list:
    logger.info(
        f"Loading {len(model_input_df)} waveform train windows from {cfg.data_path}"
    )

    def _load_single_waveform(filename: str) -> np.ndarray:
        filepath = f"{cfg.data_path}/train_windows_n5_s5_c9/{filename}"
        waveform, _ = librosa.load(filepath, sr=cfg.sample_rate)
        if normalize:
            waveform = librosa.util.normalize(waveform)

        return waveform

    waveforms = [
        _load_single_waveform(filename)
        for filename in tqdm(model_input_df["window_filename"], desc="Loading waves")
    ]

    return waveforms


def read_waveform_torch(
    filename: str, normalize: bool = cfg.normalize_waveform
) -> np.ndarray:
    filepath = f"{cfg.data_path}/train_windows_n5_s5_c9/{filename}"
    waveform, sample_rate = torchaudio.load(filepath, normalize=cfg.normalize_waveform)

    if sample_rate != cfg.sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sample_rate, new_freq=cfg.sample_rate
        )

    return waveform.squeeze()


def create_melspec_torch(waveform: torch.tensor) -> torch.tensor:
    melspec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        f_min=cfg.freq_min,
        f_max=cfg.freq_max,
        n_mels=cfg.melspec_hres,
        # power=cfg.log_scale_power,
        normalized=True,
        center=True,
        pad_mode="constant",
        mel_scale="slaney",
    )
    decibel_transform = torchaudio.transforms.AmplitudeToDB(
        stype="power", top_db=cfg.max_decibels
    )

    melspec = melspec_transform(waveform)
    melspec = decibel_transform(melspec)

    melspec = melspec - melspec.min()
    melspec = (melspec / melspec.max() * 255).type(torch.uint8)

    if melspec.shape != (128, 312):
        melspec = _pad_or_crop_melspec_torch(melspec)

    return melspec


def load_waveform_train_windows_torch(
    model_input_df: pd.DataFrame, normalize: bool = True
) -> list:
    logger.info(f"Loading {len(model_input_df)} waveform windows from {cfg.data_path}")
    waveforms = [
        read_waveform_torch(filename)
        for filename in tqdm(model_input_df["window_filename"], desc="Loading waves")
    ]

    return waveforms


def _pad_or_crop_melspec_torch(melspec: torch.tensor) -> torch.tensor:
    width = melspec.shape[-1]
    if width < cfg.melspec_wres:  # repeat if birdy melspec too small
        repeat_length = cfg.melspec_wres - width
        melspec = torch.cat([melspec, melspec[:, :repeat_length]], axis=1)
        width = melspec.shape[-1]

        if width < cfg.melspec_wres:  # repeat if birdy melspec still too small
            repeat_length = cfg.melspec_wres - width
            melspec = torch.cat([melspec, melspec[:, :repeat_length]], axis=1)
            width = melspec.shape[-1]

    if width > cfg.melspec_wres:  # crop if birdy melspec is too big
        offset = np.random.randint(0, width - cfg.melspec_wres)
        melspec = melspec[:, offset : offset + cfg.melspec_wres]

    return melspec


def read_waveforms_parallel(model_input_df: pd.DataFrame):
    logger.info("Start reading waveforms")
    waveforms = Parallel(n_jobs=cfg.n_workers)(
        delayed(read_waveform)(filename)
        for filename in tqdm(model_input_df["window_filename"], desc="Loading waves")
    )
    return waveforms


# def read_waveform(
#     filename: str, normalize: bool = cfg.normalize_waveform
# ) -> np.ndarray:
#     filepath = f"{cfg.data_path}/train_windows_n5_s5_c9/{filename}"
#     waveform, _ = librosa.load(filepath, sr=cfg.sample_rate)

#     if normalize:
#         waveform = librosa.util.normalize(waveform)

#     return waveform


# def create_melspec(waveform: np.ndarray) -> np.ndarray:
#     melspec = librosa.feature.melspectrogram(
#         y=waveform,
#         sr=cfg.sample_rate,
#         n_fft=cfg.n_fft,
#         hop_length=cfg.hop_length,
#         n_mels=cfg.melspec_hres,
#         fmin=cfg.freq_min,
#         fmax=cfg.freq_max,
#         power=cfg.log_scale_power,
#     )

#     melspec = librosa.power_to_db(melspec, ref=cfg.max_decibels)
#     melspec = melspec - melspec.min()
#     melspec = (melspec / melspec.max() * 255).astype(np.uint8)

#     if melspec.shape != (128, 312):
#         melspec = _pad_or_crop_melspec(melspec)

#     return melspec


# def _pad_or_crop_melspe(melspec: np.ndarray) -> np.ndarray:
#     width = melspec.shape[1]
#     if width < cfg.melspec_wres:  # repeat if birdy melspec too small
#         repeat_length = cfg.melspec_wres - width
#         melspec = np.concatenate([melspec, melspec[:, :repeat_length]], axis=1)
#         width = melspec.shape[1]

#         if width < cfg.melspec_wres:  # repeat if birdy melspec still too small
#             repeat_length = cfg.melspec_wres - width
#             melspec = np.concatenate([melspec, melspec[:, :repeat_length]], axis=1)
#             width = melspec.shape[1]

#     if width > cfg.melspec_wres:  # crop if birdy melspec is too big
#         offset = np.random.randint(0, width - cfg.melspec_wres)
#         melspec = melspec[:, offset : offset + cfg.melspec_wres]

#     return melspec


class BirdDataset(Dataset):
    def __init__(self, waveforms: list, labels: list, class_to_label_map: list):
        # def __init__(self, model_input_df: pd.DataFrame, class_to_label_map: list):
        self.waveforms = waveforms
        self.labels = torch.tensor(labels, dtype=torch.int64)
        # self.model_input_df = model_input_df
        self.class_to_label_map = class_to_label_map

        # self.mel_transform = torchaudio.transforms.MelSpectrogram(
        #     sample_rate=cfg.sample_rate,
        #     n_mels=cfg.melspec_hres,
        #     f_min=cfg.freq_min,
        #     f_max=cfg.freq_max,
        #     n_fft=cfg.n_fft,
        #     hop_length=cfg.hop_length,
        #     normalized=cfg.normalize_waveform,
        #     center=True,
        #     pad_mode="reflect",
        #     norm="slaney",
        #     mel_scale="slaney",
        # )
        # self.db_transform = torchaudio.transforms.AmplitudeToDB(
        #     stype="power", top_db=cfg.max_decibels
        # )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        waveform = self.waveforms[idx]
        label = self.labels[idx]

        # filename = self.model_input_df.iloc[idx]["window_filename"]
        # waveform = read_waveform(filename=filename)

        waveform = torch.tensor(waveform, dtype=torch.uint8).squeeze()
        melspec = create_melspec_torch(waveform=waveform)
        # melspec = torch.tensor(melspec, dtype=torch.uint8)

        # primary_label = self.model_input_df.iloc[idx]["primary_label"]
        # label = self.class_to_label_map.get(primary_label)
        # label = torch.tensor(label, dtype=torch.int64)

        return melspec, label


# def compute_competition_roc_score(
#     solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str
# ) -> float:
#     del solution[row_id_column_name]
#     del submission[row_id_column_name]

#     solution_sums = solution.sum(axis=0)
#     scored_columns = list(solution_sums[solution_sums > 0].index.values)
#     assert len(scored_columns) > 0

#     score = roc_auc_score(
#         solution[scored_columns].values,
#         submission[scored_columns].values,
#         average="macro",
#     )

#     return score


# def compute_padded_cmap(solution, submission, padding_factor=5):
#     solution = solution.drop(["row_id"], axis=1, errors="ignore")
#     submission = submission.drop(["row_id"], axis=1, errors="ignore")

#     new_rows = []
#     for i in range(padding_factor):
#         new_rows.append([1 for j in range(len(solution.columns))])

#     new_rows = pd.DataFrame(new_rows)
#     new_rows.columns = solution.columns

#     padded_solution = pd.concat([solution, new_rows]).reset_index(drop=True).copy()
#     padded_submission = pd.concat([submission, new_rows]).reset_index(drop=True).copy()

#     score = average_precision_score(
#         padded_solution.values.astype(int),
#         padded_submission.values,
#         average="macro",
#     )
#     return score


# def compute_competition_metrics(gt, preds, target_columns, one_hot=True):
#     if not one_hot:
#         ground_truth = np.argmax(gt, axis=1)
#         gt = np.zeros((ground_truth.size, len(target_columns)))
#         gt[np.arange(ground_truth.size), ground_truth] = 1

#     val_df = pd.DataFrame(gt, columns=target_columns)
#     pred_df = pd.DataFrame(preds, columns=target_columns)

#     cmap_1 = compute_padded_cmap(val_df, pred_df, padding_factor=1)
#     cmap_5 = compute_padded_cmap(val_df, pred_df, padding_factor=5)

#     val_df["id"] = [f"id_{i}" for i in range(len(val_df))]
#     pred_df["id"] = [f"id_{i}" for i in range(len(pred_df))]

#     competition_score = compute_competition_roc_score(
#         val_df, pred_df, row_id_column_name="id"
#     )
#     return {
#         "cmAP_1": cmap_1,
#         "cmAP_5": cmap_5,
#         "Comp ROC": competition_score,
#     }


# def metrics_to_string(scores, key_word):
#     log_info = ""
#     for key in scores.keys():
#         log_info = log_info + f"{key_word} {key} : {scores[key]:.4f}, "
#     return log_info


if __name__ == "__main__":
    logger = define_logger()
    config_dictionary = get_config(cfg)

    model_input_df, sample_submission = load_metadata(
        data_path=cfg.data_path, dry_run=cfg.dry_run
    )
    waveforms = load_waveform_train_windows(model_input_df, normalize=False)

    class_to_label_map = create_label_map(submission_df=sample_submission)
    labels = [
        class_to_label_map.get(primary_label)
        for primary_label in tqdm(
            model_input_df["primary_label"], desc="Generating train wave labels"
        )
    ]

    # waveforms = load_windowed_waveforms(model_input_df, normalize=False)

    # df_train, df_val, y_train, y_val = train_test_split(
    #     model_input_df,
    #     test_size=cfg.val_ratio,
    #     stratify=model_input_df["primary_label"],
    #     shuffle=True,
    #     random_state=None,
    # )

    n_splits = int(round(1 / cfg.val_ratio))
    kfold = StratifiedGroupKFold(n_splits=n_splits, shuffle=True)
    for train_index, val_index in kfold.split(
        X=model_input_df,
        y=model_input_df["primary_label"],
        groups=model_input_df["sample_index"],
    ):
        break

    train_waves = [waveforms[i] for i in train_index]
    train_labels = [labels[i] for i in train_index]

    val_waves = [waveforms[i] for i in val_index]
    val_labels = [labels[i] for i in val_index]

    train_dataset = BirdDataset(train_waves, train_labels, class_to_label_map)
    val_dataset = BirdDataset(val_waves, val_labels, class_to_label_map)

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

            self.loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)
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
            loss = self.loss_function(y_pred, y)

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

            val_loss = self.loss_function(y_pred, y_val)
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

        def on_train_epoch_end(self):
            metrics = self.trainer.progress_bar_callback.get_metrics(trainer, model)
            metrics.pop("v_num", None)
            for key, value in metrics.items():
                metrics[key] = round(value, 4)
            logger.info(f"Epoch {self.trainer.current_epoch}: {metrics}")

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

    csv_logger = None
    if not cfg.dry_run:
        csv_logger = L.pytorch.loggers.CSVLogger(save_dir="../logs/")
        csv_logger.log_hyperparams(config_dictionary)

    model = EfficientViT()
    trainer = L.Trainer(
        fast_dev_run=False,
        enable_model_summary=True,
        max_epochs=cfg.n_epochs,
        accelerator=cfg.accelerator,
        precision=cfg.precision,
        callbacks=[TQDMProgressBar()],
        logger=csv_logger,
        log_every_n_steps=10,
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=None,
    )

    logger.info("Finished training, saving model")
    cfg.timestamp = datetime.now().replace(microsecond=0)
    # trainer.save_checkpoint(f"model_objects/full_{cfg.run_tag}.ckpt")
    logger.info("All done")
