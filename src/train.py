import gc
import logging
import os
import random
import warnings
from datetime import datetime
from pprint import pformat

import albumentations
import librosa
import lightning as L
import matplotlib.pyplot as plt
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
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2 as transforms
from tqdm.notebook import tqdm

from losses import FocalCosineLoss, FocalLoss, FocalLossBCE, JsdCrossEntropy
from optimizers import Adan, Nadam, NvidiaNovoGrad

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
    experiment_name = "efnetv2s"
    data_path = "../data"

    debug_run = False  # run on a small sample
    experiment_run = False  # run on a stratified data sample
    production_run = True  # run on all data

    normalize_waveform = False  # TODO test with and without
    sample_rate = 32000
    n_fft = 2048
    hop_length = 512
    window_length = None
    melspec_hres = 128
    melspec_wres = 312
    freq_min = 40
    freq_max = 15000
    log_scale_power = 2
    max_decibels = 100
    frame_duration = 5
    frame_rate = sample_rate / hop_length

    # vit_b0 = "efficientvit_b0.r224_in1k"
    # vit_b1 = "efficientvit_b1.r224_in1k"
    # vit_b1 = "efficientvit_b1.r288_in1k"
    # vit_b2 = "efficientvit_b2.r224_in1k"
    # efnet_b0 = "tf_efficientnet_b0.in1k"
    # efnetv2_b0 = "tf_efficientnetv2_b0.in1k"
    efnetv2s = "tf_efficientnetv2_s.in21k_ft_in1k"  # TODO
    # efnet_b0_jft = "tf_efficientnet_b0.ns_jft_in1k" # TODO
    # efnetv2_b1 = "tf_efficientnetv2_b1.in1k"
    # vit_m0 = "efficientvit_m0.r224_in1k"
    # vit_m1 = "efficientvit_m1.r224_in1k"
    backbone = efnetv2s

    num_classes = 182
    mixup = True
    mixup_alpha = 3
    augment_melspec = True
    add_secondary_labels = False
    add_secondary_label_weight = 0.33

    label_smoothing = 0.1
    weighted_sampling = False
    sample_weight_factor = 0.5

    accelerator = "gpu"
    precision = "bf16-mixed"
    n_workers = os.cpu_count() - 2

    n_epochs = 50
    batch_size = 128
    val_ratio = 0.20
    patience = 10

    lr = 1e-3
    lr_min = 1e-6
    weight_decay = 1e-3

    loss = "FocalBCE"
    optimizer = "AdamW"

    timestamp = datetime.now().replace(microsecond=0)
    run_tag = f"{timestamp}_{backbone}_{experiment_name}_val_{val_ratio}_{loss}_{optimizer}_lr_{lr}_decay_{weight_decay}"

    if debug_run:
        run_tag = f"{timestamp}_{backbone}_debug"
        # accelerator = "cpu"
        n_epochs = 5
        batch_size = 32
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
        model_input_df = model_input_df.sample(1000).reset_index(drop=True)

    elif cfg.experiment_run:
        logger.info("Running experiment: sampling data")
        model_input_df = model_input_df.sample(frac=0.1).reset_index(drop=True)

    elif cfg.production_run:
        logger.info("Running production: full data")
        model_input_df = model_input_df.sample(frac=1, random_state=7).reset_index(
            drop=True
        )

    logger.info(f"Dataframe shape: {model_input_df.shape}")

    return model_input_df, sample_submission


def undersample_top_birds(
    model_input_df: pd.DataFrame, top_k: int = 20
) -> pd.DataFrame:
    top_k_birds = model_input_df["primary_label"].value_counts()[0:top_k].index
    top_k_birds_idx = model_input_df[model_input_df["primary_label"].isin(top_k_birds)][
        ["sample_index", "primary_label"]
    ].index

    idx_to_keep = (
        model_input_df[model_input_df["primary_label"].isin(top_k_birds)][
            ["sample_index", "primary_label"]
        ]
        .drop_duplicates()
        .index
    )
    idx_to_drop = top_k_birds_idx[~top_k_birds_idx.isin(idx_to_keep)]

    model_input_df = model_input_df[~model_input_df.index.isin(idx_to_drop)]
    logger.info(f"Dataframe shape after undersampling: {model_input_df.shape}")

    return model_input_df


def add_sample_weights(
    model_input_df: pd.DataFrame, weight_factor: float = cfg.sample_weight_factor
) -> pd.DataFrame:
    sample_weights = round(
        (
            model_input_df["primary_label"].value_counts()
            / model_input_df["primary_label"].value_counts().sum()
        )
        ** (-weight_factor)
    )
    sample_weights = pd.DataFrame(
        {
            "primary_label": sample_weights.index,
            "sample_weight": sample_weights.values.astype(int),
        }
    )
    model_input_df = model_input_df.merge(
        sample_weights, on="primary_label", how="left"
    )
    return model_input_df


def read_waveform(filename: str) -> np.ndarray:
    filepath = f"{cfg.data_path}/train_windows/{filename}"
    waveform, _ = librosa.load(filepath, sr=cfg.sample_rate)

    if cfg.normalize_waveform:
        waveform = librosa.util.normalize(waveform)

    return waveform


def read_waveforms_parallel(model_input_df: pd.DataFrame):
    logger.info("Parallel Loading waveforms")
    waveforms = Parallel(n_jobs=cfg.n_workers, prefer="threads")(
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


def pad_or_crop_waveforms(waveforms: list, pad_method: str = "repeat") -> list:
    logging.info("Padding or cropping waveforms to desired duration")
    desired_length = cfg.sample_rate * cfg.frame_duration

    def _pad_or_crop(waveform: np.ndarray) -> np.ndarray:
        length = len(waveform)

        while length < desired_length:  # repeat if waveform too small
            repeat_length = desired_length - length
            padding_array = waveform[:repeat_length]
            if pad_method != "repeat":
                padding_array = np.zeros(shape=waveform[:repeat_length].shape)
            waveform = np.concatenate([waveform, padding_array])
            length = len(waveform)

        if length > desired_length:  # crop if waveform is too big
            offset = np.random.randint(0, length - desired_length)
            waveform = waveform[offset : offset + desired_length]

        return waveform

    waveforms = [_pad_or_crop(wave) for wave in tqdm(waveforms, desc="Padding waves")]

    return waveforms


class BirdDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        waveforms: list,
        add_secondary_labels: bool = cfg.add_secondary_labels,
        augmentation: list = None,
    ):
        self.df = df
        self.waveforms = waveforms
        self.num_classes = cfg.num_classes
        self.class_to_label_map = class_to_label_map
        self.add_secondary_labels = add_secondary_labels
        self.augmentation = augmentation

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
        secondary_label_weight: float = cfg.add_secondary_label_weight,
    ) -> torch.tensor:
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        # primary_target = torch.tensor(0, dtype=torch.int64)

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

    def pad_or_crop_wave(
        self, waveform: np.ndarray, pad_method: str = "repeat"
    ) -> np.ndarray:
        desired_length = cfg.sample_rate * cfg.frame_duration
        length = len(waveform)

        while length < desired_length:  # repeat if waveform too small
            repeat_length = desired_length - length
            padding_array = waveform[:repeat_length]
            if pad_method != "repeat":
                padding_array = np.zeros(shape=waveform[:repeat_length].shape)
            waveform = np.concatenate([waveform, padding_array])
            length = len(waveform)

        if length > desired_length:  # crop if waveform is too big
            offset = np.random.randint(0, length - desired_length)
            waveform = waveform[offset : offset + desired_length]

        return waveform

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        waveform = self.waveforms[idx]
        primary_label = self.df.iloc[idx]["primary_label"]
        secondary_labels = self.df.iloc[idx]["secondary_labels"]

        waveform = self.pad_or_crop_wave(waveform)
        waveform = torch.tensor(waveform, dtype=torch.float32)

        target, primary_target = self.create_target(
            primary_label=primary_label, secondary_labels=secondary_labels
        )

        melspec = self.db_transform(self.mel_transform(waveform)).to(torch.uint8)
        melspec = melspec.expand(3, -1, -1).permute(1, 2, 0).numpy()

        melspec = melspec - melspec.min()
        melspec = melspec / melspec.max()
        melspec = melspec.astype(np.float32)

        if self.augmentation is not None:
            melspec = self.augmentation(image=melspec)["image"]

        return melspec, target, primary_target


class GlobalPool(torch.nn.Module):
    def __init__(self, p: int = 3, eps: float = 1e-6):
        super(GlobalPool, self).__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        bs, ch, h, w = x.shape
        x = torch.nn.functional.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)
        x = x.view(bs, ch)
        return x


class EfficientNetV2(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.normalize = transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )

        out_indices = (3, 4)
        self.backbone = timm.create_model(
            cfg.backbone,
            features_only=True,
            pretrained=True,
            in_chans=3,
            num_classes=cfg.num_classes,
            out_indices=out_indices,
        )
        self.in_features = int(np.sum(self.backbone.feature_info.channels()))
        print(f"in_features: {self.in_features}")

        self.global_pools = torch.nn.ModuleList([GlobalPool() for _ in out_indices])
        self.neck = torch.nn.BatchNorm1d(self.in_features)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.in_features, out_features=256),
            torch.nn.Hardswish(inplace=True),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(in_features=256, out_features=cfg.num_classes),
        )

        # self.loss_function = FocalCosineLoss()
        self.loss_function = FocalLossBCE()
        # self.loss_function = FocalLoss()
        # self.loss_function = torch.nn.BCEWithLogitsLoss(reduction="mean")
        # self.loss_function = torch.nn.CrossEntropyLoss(
        #     label_smoothing=cfg.label_smoothing
        # )

        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=cfg.num_classes, top_k=1
        )
        self.accuracy_2 = torchmetrics.Accuracy(
            task="multiclass", num_classes=cfg.num_classes, top_k=2
        )
        self.auroc = torchmetrics.AUROC(
            task="multilabel",
            num_labels=cfg.num_classes,
            average="macro",
        )
        self.f1_macro = torchmetrics.F1Score(
            task="multilabel",
            num_labels=cfg.num_classes,
            average="macro",
            threshold=0.5,
        )
        self.f1_weighted = torchmetrics.F1Score(
            task="multilabel",
            num_labels=cfg.num_classes,
            average="weighted",
            threshold=0.5,
        )
        self.lrap = torchmetrics.classification.MultilabelRankingAveragePrecision(
            num_labels=cfg.num_classes,
        )
        self.precision_macro = torchmetrics.classification.MultilabelPrecision(
            num_labels=cfg.num_classes,
            average="macro",
            threshold=0.5,
        )
        self.precision_weighted = torchmetrics.classification.MultilabelPrecision(
            num_labels=cfg.num_classes,
            average="weighted",
            threshold=0.5,
        )
        self.recall_macro = torchmetrics.classification.MultilabelRecall(
            num_labels=cfg.num_classes,
            average="macro",
            threshold=0.5,
        )
        self.recall_weighted = torchmetrics.classification.MultilabelRecall(
            num_labels=cfg.num_classes,
            average="weighted",
            threshold=0.5,
        )

    def normal_mixup(self, melspec, target, alpha=cfg.mixup_alpha):
        indices = torch.randperm(melspec.size(0))
        mix_melspec = melspec[indices]
        mix_target = target[indices]

        lam = np.random.beta(alpha, alpha)
        if lam < 0.5:
            lam = 1 - lam

        melspec = melspec * lam + mix_melspec * (1 - lam)
        target = target * lam + mix_target * (1 - lam)

        return melspec, target

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.normalize(x)
        x = self.backbone(x)
        pooled_features = [self.global_pools[0](x[0]), self.global_pools[1](x[1])]
        x = torch.cat(pooled_features, dim=1)
        x = self.neck(x)
        out = self.head(x)

        return out

    def training_step(self, batch, batch_idx):
        x, y, y_primary = batch

        if cfg.mixup:
            choice = np.random.choice(["normal", "none"])
            if choice == "normal":
                x, y = self.normal_mixup(x, y)

        y_pred = self(x)
        loss = self.loss_function(y_pred, y)

        y_pred = y_pred.sigmoid()
        y_int = y.to(torch.int64)

        train_accuracy = self.accuracy(y_pred, y_primary)
        train_accuracy_2 = self.accuracy_2(y_pred, y_primary)
        train_f1_weighted = self.f1_weighted(y_pred, y_int)
        train_f1_macro = self.f1_macro(y_pred, y_int)
        train_lrap = self.lrap(y_pred, y_int)

        train_precision_macro = self.precision_macro(y_pred, y_int)
        train_precision_weighted = self.precision_weighted(y_pred, y_int)
        train_recall_macro = self.recall_macro(y_pred, y_int)
        train_recall_weighted = self.recall_weighted(y_pred, y_int)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_accuracy",
            train_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_accuracy_2",
            train_accuracy_2,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_f1_weighted",
            train_f1_weighted,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_f1_macro",
            train_f1_macro,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_lrap",
            train_lrap,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_precision_macro",
            train_precision_macro,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_precision_weighted",
            train_precision_weighted,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_recall_macro",
            train_recall_macro,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_recall_weighted",
            train_recall_weighted,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x_val, y_val, y_primary_val = batch
        y_pred = self(x_val)
        val_loss = self.loss_function(y_pred, y_val)

        y_pred = y_pred.sigmoid()
        y_val_int = y_val.to(torch.int64)

        val_accuracy = self.accuracy(y_pred, y_primary_val)
        val_accuracy_2 = self.accuracy_2(y_pred, y_primary_val)
        val_auroc = self.auroc(y_pred, y_val_int)
        val_f1_weighted = self.f1_weighted(y_pred, y_val_int)
        val_f1_macro = self.f1_macro(y_pred, y_val_int)
        val_lrap = self.lrap(y_pred, y_val_int)

        val_precision_macro = self.precision_macro(y_pred, y_val_int)
        val_precision_weighted = self.precision_weighted(y_pred, y_val_int)
        val_recall_macro = self.recall_macro(y_pred, y_val_int)
        val_recall_weighted = self.recall_weighted(y_pred, y_val_int)

        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_accuracy",
            val_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_accuracy_2",
            val_accuracy_2,
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
        self.log(
            "val_f1_weighted",
            val_f1_weighted,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_f1_macro",
            val_f1_macro,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_lrap",
            val_lrap,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_precision_macro",
            val_precision_macro,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_precision_weighted",
            val_precision_weighted,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_recall_macro",
            val_recall_macro,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_recall_weighted",
            val_recall_weighted,
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
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            fused=False,
        )
        lr_scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.n_epochs, T_mult=1, eta_min=cfg.lr_min, last_epoch=-1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "monitor": "train_lrap",
                "frequency": 1,
            },
        }


if __name__ == "__main__":
    logger = define_logger()
    config_dictionary = get_config(cfg)

    csv_logger = None
    if not cfg.debug_run:
        csv_logger = L.pytorch.loggers.CSVLogger(save_dir="../logs/")
        csv_logger.log_hyperparams(config_dictionary)

    model_input_df, sample_submission = load_metadata(data_path=cfg.data_path)
    model_input_df = undersample_top_birds(model_input_df)
    model_input_df = add_sample_weights(model_input_df)
    class_to_label_map = create_label_map(submission_df=sample_submission)

    waveforms = read_waveforms_parallel(model_input_df=model_input_df)

    if cfg.augment_melspec:
        train_augmentation = albumentations.Compose(
            [
                albumentations.AdvancedBlur(p=0.20),
                albumentations.GaussNoise(p=0.20),
                albumentations.ImageCompression(
                    quality_lower=80, quality_upper=100, p=0.20
                ),
                albumentations.CoarseDropout(
                    max_holes=1,
                    min_height=16,
                    min_width=16,
                    max_height=48,
                    max_width=48,
                    p=0.20,
                ),
                albumentations.XYMasking(
                    p=0.20,
                    num_masks_x=1,
                    num_masks_y=1,
                    mask_x_length=(3, 20),
                    mask_y_length=(3, 20),
                ),
            ]
        )
    else:
        train_augmentation = None

    logger.info(f"Splitting {len(waveforms)} waveforms into train/val: {cfg.val_ratio}")
    n_splits = int(round(1 / cfg.val_ratio))
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
    for fold_index, (train_index, val_index) in enumerate(
        kfold.split(
            X=model_input_df,
            y=model_input_df["primary_label"],
            groups=model_input_df["sample_index"],
        )
    ):

        train_df = model_input_df.iloc[train_index]
        val_df = model_input_df.iloc[val_index]

        train_waveforms = [waveforms[i] for i in train_index]
        val_waveforms = [waveforms[i] for i in val_index]

        train_dataset = BirdDataset(
            df=train_df, waveforms=train_waveforms, augmentation=train_augmentation
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

        if cfg.weighted_sampling:
            logger.info(
                f"Defining weighted sampling with  factor: {cfg.sample_weight_factor}"
            )
            sample_weight = train_df["sample_weight"].values
            sample_weight = torch.from_numpy(sample_weight)

            weighted_sampler = WeightedRandomSampler(
                sample_weight.type("torch.DoubleTensor"),
                len(sample_weight),
                replacement=True,
            )

            train_dataloader = DataLoader(
                train_dataset,
                batch_size=cfg.batch_size,
                sampler=weighted_sampler,
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

        logger.info("Dataloaders ready to go brrr")

        progress_bar = TQDMProgressBar(process_position=1)
        early_stopping = EarlyStopping(
            monitor="train_lrap",
            min_delta=0.01,
            patience=cfg.patience,
            verbose=True,
            mode="max",
        )
        model_checkpoint = ModelCheckpoint(
            monitor="train_lrap",
            every_n_epochs=1,
            mode="max",
            auto_insert_metric_name=True,
            filename=f"{cfg.run_tag}"
            + f"_fold_{fold_index}_"
            + "{epoch}-{val_lrap:.3f}-{val_acc2:.3f}-{val_acc2:.3f}-{val_f1_macro:.3}",
            dirpath="../model_objects/ckpts/",
        )

        os.environ["PJRT_DEVICE"] = "GPU"  # fix for G Cloud to avoid XLA/autocast clash
        model = EfficientNetV2()
        trainer = L.Trainer(
            fast_dev_run=False,
            enable_model_summary=True,
            max_epochs=cfg.n_epochs,
            accelerator=cfg.accelerator,
            precision=cfg.precision,
            callbacks=[progress_bar, early_stopping, model_checkpoint],
            logger=csv_logger,
            log_every_n_steps=10,
        )

        logger.info(f"\nStart training fold {fold_index}")
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=None,
        )

        logger.info(f"Finished training fold {fold_index}")
        if not cfg.debug_run and trainer.current_epoch > 10:
            logger.info("Saving model")
            filename = (
                f"{cfg.run_tag}_fold_{fold_index}_epochs_{trainer.current_epoch}.ckpt"
            )
            trainer.save_checkpoint(f"../model_objects/{filename}")
            logger.info(f"Saved model to filename: {filename}")
