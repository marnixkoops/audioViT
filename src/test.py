# %%
# %%
# !pip install lightning timm librosa albumentations torchaudio==2.2.2 audiomentations


# %%
# %%
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
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
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


# %%
# %%
class cfg:
    experiment_name = "base"
    data_path = "../data"

    debug_run = True  # run on a small sample
    experiment_run = False  # run on a stratified data sample
    production_run = False  # run on all data

    normalize_waveform = True
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
    efnetv2_b0 = "tf_efficientnetv2_b0.in1k"
    # efnetv2s = "tf_efficientnetv2_s.in21k_ft_in1k" # TODO
    # efnetv2_b1 = "tf_efficientnetv2_b1.in1k"
    # vit_m0 = "efficientvit_m0.r224_in1k"
    # vit_m1 = "efficientvit_m1.r224_in1k"
    backbone = efnetv2_b0

    num_classes = 182
    add_secondary_labels = False
    mixup_prob = 0.0
    label_smoothing = 0.0
    weighted_sampling = False
    sample_weight_factor = 0.25

    accelerator = "gpu"
    precision = "16-mixed"
    n_workers = os.cpu_count() - 2

    n_epochs = 50
    batch_size = 128
    val_ratio = 0.2
    patience = 10

    lr = 1e-4
    lr_min = 1e-6
    weight_decay = 0.0

    loss = "CrossEntropyLoss"
    optimizer = "Adan"

    timestamp = datetime.now().replace(microsecond=0)
    run_tag = f"{timestamp}_{backbone}_{experiment_name}_val_{val_ratio}_lr_{lr}_decay_{weight_decay}"

    if debug_run:
        run_tag = f"{timestamp}_{backbone}_debug"
        accelerator = "cpu"
        n_epochs = 5
        batch_size = 32
        fused_adamw = False
        num_classes = 10


# %%
# %%
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


def read_waveform(filename: str) -> np.ndarray:
    filepath = f"{cfg.data_path}/train_windows_nv_c10/{filename}"
    waveform, _ = librosa.load(filepath, sr=cfg.sample_rate)
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


# %%
# %%
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
        secondary_label_weight: float = 1,
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

        binary_target = target.to(torch.int64)

        return target, binary_target, primary_target

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        waveform = self.waveforms[idx]
        primary_label = self.df.iloc[idx]["primary_label"]
        secondary_labels = self.df.iloc[idx]["secondary_labels"]

        waveform = torch.tensor(waveform, dtype=torch.float32)

        target, binary_target, primary_target = self.create_target(
            primary_label=primary_label, secondary_labels=secondary_labels
        )

        melspec = self.db_transform(self.mel_transform(waveform)).to(torch.uint8)
        melspec = melspec.expand(3, -1, -1).permute(1, 2, 0).numpy()

        if self.augmentation is not None:
            melspec = self.augmentation(image=melspec)["image"]

        return melspec, target, binary_target, primary_target


# %%
# %%


class GeM(torch.nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
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

        out_indices = (3, 4)
        self.efnetv2 = timm.create_model(
            cfg.backbone,
            features_only=True,
            pretrained=True,
            in_chans=3,
            num_classes=cfg.num_classes,
            out_indices=out_indices,
        )

        feature_dims = self.efnetv2.feature_info.channels()
        logger.info(f"EfficientNetV2 feature dims: {feature_dims}")

        self.global_pools = torch.nn.ModuleList([GeM() for _ in out_indices])
        self.mid_features = np.sum(feature_dims)
        self.neck = torch.nn.BatchNorm1d(self.mid_features)
        self.head = torch.nn.Linear(self.mid_features, cfg.num_classes)

        # self.loss_function = FocalLossBCE()
        # self.loss_function = FocalLoss()
        # self.loss_function = torch.nn.BCEWithLogitsLoss(reduction="mean")
        self.loss_function = torch.nn.CrossEntropyLoss(
            label_smoothing=cfg.label_smoothing
        )

        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=cfg.num_classes, top_k=1
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

    def mixup(self, data, targets, alpha):
        indices = torch.randperm(data.size(0))
        data2 = data[indices]
        targets2 = targets[indices]

        lam = torch.FloatTensor([np.random.beta(alpha, alpha)])
        data = data * lam + data2 * (1 - lam)
        targets = targets * lam + targets2 * (1 - lam)

        return data, targets

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.efnetv2(x)
        x = torch.cat(
            [global_pool(m) for m, global_pool in zip(x, self.global_pools)], dim=1
        )
        x = self.neck(x)
        out = self.head(x)

        return out

    def training_step(self, batch, batch_idx):
        x, y, y_binary, y_primary = batch

        if cfg.mixup_prob > 0:
            if np.random.choice([True, False]):
                x, y = self.mixup(x, y, 0.5)

        y_pred = self(x)
        loss = self.loss_function(y_pred, y)
        y_pred = y_pred.sigmoid()

        train_accuracy = self.accuracy(y_pred, y_primary)
        train_f1_weighted = self.f1_weighted(y_pred, y_binary)
        train_f1_macro = self.f1_macro(y_pred, y_binary)
        train_lrap = self.lrap(y_pred, y_binary)

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

        return loss

    def validation_step(self, batch, batch_idx):
        x_val, y_val, y_binary_val, y_primary_val = batch
        y_pred = self(x_val)
        val_loss = self.loss_function(y_pred, y_val)

        y_pred = y_pred.sigmoid()
        val_accuracy = self.accuracy(y_pred, y_primary_val)
        val_auroc = self.auroc(y_pred, y_binary_val)
        val_f1_weighted = self.f1_weighted(y_pred, y_binary_val)
        val_f1_macro = self.f1_macro(y_pred, y_binary_val)
        val_lrap = self.lrap(y_pred, y_binary_val)

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

        return val_loss

    def on_train_epoch_end(self):
        metrics = self.trainer.progress_bar_callback.get_metrics(trainer, model)
        metrics.pop("v_num", None)
        metrics.pop("train_loss_step", None)
        for key, value in metrics.items():
            metrics[key] = round(value, 5)
        logger.info(f"Epoch {self.trainer.current_epoch}: {metrics}")

    def configure_optimizers(self):
        optimizer = Adan(
            model.parameters(),
            lr=cfg.lr,
            betas=(0.02, 0.08, 0.01),
            weight_decay=cfg.weight_decay,
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


# %%
# %%
if __name__ == "__main__":
    logger = define_logger()
    config_dictionary = get_config(cfg)

    csv_logger = None
    if not cfg.debug_run:
        csv_logger = L.pytorch.loggers.CSVLogger(save_dir="../logs/")
        csv_logger.log_hyperparams(config_dictionary)

    model_input_df, sample_submission = load_metadata(data_path=cfg.data_path)
    model_input_df = add_sample_weights(model_input_df)
    class_to_label_map = create_label_map(submission_df=sample_submission)

    # %%
    # %%
    waveforms = read_waveforms_parallel(model_input_df=model_input_df)

    # %%
    # %%
    waveforms = pad_or_crop_waveforms(waveforms=waveforms)

    # %%
    # %%
    train_augmentation = albumentations.Compose(
        [
            # albumentations.AdvancedBlur(p=0.25),
            # albumentations.GaussNoise(p=0.25),
            albumentations.ImageCompression(quality_lower=90, quality_upper=100, p=1.0),
            # albumentations.CoarseDropout(
            #     max_holes=1, max_height=64, max_width=64, p=0.25
            # ),
            # albumentations.XYMasking(
            #     p=0.25,
            #     num_masks_x=(1, 2),
            #     num_masks_y=(1, 2),
            #     mask_x_length=(5, 25),
            #     mask_y_length=(5, 25),
            # ),
            # albumentations.CLAHE(p=0.15),
            # albumentations.Downscale(
            #     scale_min=0.5, scale_max=0.9, interpolation=4, p=0.15
            # ),
            # albumentations.Morphological(p=0.15, scale=(1, 3), operation="erosion"),
            # albumentations.Normalize(p=1),
        ]
    )
    val_augmentation = albumentations.Compose(
        [albumentations.ImageCompression(quality_lower=90, quality_upper=100, p=1.0)]
    )

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
        break

    train_df = model_input_df.iloc[train_index]
    val_df = model_input_df.iloc[val_index]

    train_waveforms = [waveforms[i] for i in train_index]
    val_waveforms = [waveforms[i] for i in val_index]

    train_dataset = BirdDataset(
        df=train_df, waveforms=train_waveforms, augmentation=train_augmentation
    )
    val_dataset = BirdDataset(
        df=val_df, waveforms=val_waveforms, augmentation=val_augmentation
    )

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

    # %%
    # %%
    progress_bar = TQDMProgressBar(process_position=1)
    early_stopping = EarlyStopping(
        monitor="val_f1_macro",
        min_delta=0.005,
        patience=cfg.patience,
        verbose=True,
        mode="max",
    )
    model_checkpoint = ModelCheckpoint(
        monitor="val_f1_macro",
        every_n_epochs=1,
        mode="max",
        save_on_train_epoch_end=True,
        auto_insert_metric_name=True,
        filename=f"{cfg.run_tag}"
        + f"_fold_{fold_index}_"
        + "{epoch}-{val_lrap:.3f}-{val_acc2:.3f}-{val_acc2:.3f}-{val_f1_macro:.3}",
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


# %%
