import gc
import pickle
from datetime import datetime

import albumentations
import lightning as L
import numpy as np
import pandas as pd
import seaborn as sns
import timm
import torch
import torchmetrics
import torchvision
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm.notebook import tqdm

# import logging
# logger = logging.getLogger(__name__)
# logging.basicConfig(
#     level=logging.DEBUG,
#     format=" %(asctime)s [%(threadName)s] [%(levelname)s] 🦤 %(message)s",
# )


sns.set(
    rc={
        "figure.figsize": (8, 6),
        "figure.dpi": 240,
    }
)
sns.set_style("whitegrid", {"axes.grid": False})


class cfg:
    # root_folder = "/kaggle/input/birdclef-2024"
    root_folder = "input/birdclef-2024"
    seed = 7

    # vit_b0 = "efficientvit_b0.r224_in1k"
    vit_b1 = "efficientvit_b1.r288_in1k"
    # effnet_b0 = "tf_efficientnetv2_b0.in1k"
    # effnet_b1 = "tf_efficientnetv2_b1.in1k"
    # vit_m0 = "efficientvit_m0.r224_in1k"
    # vit_m1 = "efficientvit_m1.r224_in1k"
    backbone = vit_b1

    n_epochs = 50
    lr_max = 1e-4
    weight_decay = 1e-6

    accelerator = "gpu"
    precision = "16-mixed"
    batch_size = 256
    n_workers = 6

    val_ratio = 0.25
    mixup_prob = 0.66
    n_classes = 182

    timestamp = datetime.now().replace(microsecond=0)
    run_tag = f"{timestamp}_{backbone}_val_{val_ratio}_lr_{lr_max}_decay_{weight_decay}_mixup_{mixup_prob}"


def load_data(path: str):
    print(f"Loading prepared data from {cfg.root_folder}")
    sample_submission = pd.read_csv(f"{cfg.root_folder}/sample_submission.csv")

    model_input_df = pd.read_csv(f"{path}/model_input_df.csv")
    with open(f"{path}/melspec_list.pkl", "rb") as file:
        melpec_list = pickle.load(file)

    with open(f"{path}/label_list.pkl", "rb") as file:
        label_list = pickle.load(file)

    cfg.labels = sample_submission.columns[1:]
    cfg.n_classes = len(cfg.labels)

    return (
        model_input_df,
        sample_submission,
        melpec_list,
        label_list,
    )


class BirdDataset(Dataset):
    def __init__(
        self,
        X,
        y,
        agumentation=None,
    ):
        self.X = X
        self.y = torch.tensor(y, dtype=torch.int64)
        self.agumentation = agumentation

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        melspec = self.X[idx]
        label = self.y[idx]

        label_ohe = torch.zeros(cfg.n_classes)
        label_ohe[label] = 1

        if self.agumentation is not None:
            melspec = self.agumentation(image=melspec)["image"]

        return melspec, label, label_ohe


if __name__ == "__main__":
    model_input_df, sample_submission, melspec_list, label_list = load_data(
        # path="/kaggle/input/2024-deduped-2104"
        path="input/birdclef-2024/model_input"
    )

    X_train, X_test, y_val, y_val = train_test_split(
        melspec_list,
        label_list,
        test_size=cfg.val_ratio,
        stratify=label_list,
        shuffle=True,
        random_state=cfg.seed,
    )

    # n_splits = int(round(1 / cfg.val_ratio))
    # kfold = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
    # for train_index, val_index in kfold.split(
    #     X=model_input_df, y=label_list, groups=model_input_df["sample_index"]
    # ):
    #     break

    # X_train = [melspec_list[i] for i in train_index]
    # X_val = [melspec_list[i] for i in val_index]

    # y_train = [label_list[i] for i in train_index]
    # y_val = [label_list[i] for i in val_index]

    train_augmentations = albumentations.Compose(
        [
            # albumentations.GaussNoise(var_limit=5.0 / 255.0, p=0.25),
            # albumentations.ImageCompression(quality_lower=80, quality_upper=100, p=0.25),
            albumentations.CoarseDropout(
                max_holes=4, max_height=20, max_width=20, p=0.25
            ),
            albumentations.XYMasking(
                p=0.1,
                num_masks_x=(1, 3),
                num_masks_y=(1, 3),
                mask_x_length=(2, 7),
                mask_y_length=(2, 7),
            ),
        ]
    )

    train_dataset = BirdDataset(X_train, y_train, agumentation=train_augmentations)
    val_dataset = BirdDataset(X_val, y_val, agumentation=None)

    sample_weight = model_input_df.loc[train_index]["sample_weight"].values
    sample_weight = torch.from_numpy(sample_weight)
    weighted_sampler = WeightedRandomSampler(
        sample_weight.type("torch.DoubleTensor"), len(sample_weight), replacement=True
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=weighted_sampler,
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

    del (
        model_input_df,
        X_train,
        X_val,
        y_train,
        y_val,
        train_index,
        val_index,
        melspec_list,
        label_list,
    )
    gc.collect()

    class FocalLoss(torch.nn.Module):
        def __init__(
            self,
            alpha: float = 0.25,
            gamma: float = 2,
            reduction: str = "mean",
        ):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, x, y):
            loss = torchvision.ops.focal_loss.sigmoid_focal_loss(
                inputs=x,
                targets=y,
                alpha=self.alpha,
                gamma=self.gamma,
                reduction=self.reduction,
            )
            return loss

    class FocalLossBCE(torch.nn.Module):
        def __init__(
            self,
            alpha: float = 0.25,
            gamma: float = 2,
            reduction: str = "none",  # or "mean"
            bce_weight: float = 1.0,
            focal_weight: float = 1.0,
        ):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction
            self.bce = torch.nn.BCEWithLogitsLoss(reduction=reduction)
            self.bce_weight = bce_weight
            self.focal_weight = focal_weight

        def forward(self, inputs, targets):
            focall_loss = torchvision.ops.focal_loss.sigmoid_focal_loss(
                inputs=inputs,
                targets=targets,
                alpha=self.alpha,
                gamma=self.gamma,
                reduction=self.reduction,
            )
            bce_loss = self.bce(inputs, targets)
            combined_loss = self.bce_weight * bce_loss + self.focal_weight * focall_loss
            return combined_loss

    class EfficientViT(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.normalize = transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            )

            self.vit = timm.create_model(
                cfg.backbone,
                pretrained=True,
                num_classes=cfg.n_classes,
            )

            self.loss_criterion = nn.CrossEntropyLoss()
            # self.loss_criterion = nn.BCEWithLogitsLoss(reduction="none")
            # self.loss_criterion = FocalLoss()
            self.loss_criterion = FocalLossBCE(reduction="none")

            self.acc = torchmetrics.Accuracy(
                task="multiclass", num_classes=cfg.n_classes
            )
            self.auroc = torchmetrics.AUROC(
                task="multiclass", num_classes=cfg.n_classes
            )

            self.checkpoint = ModelCheckpoint(
                monitor="val_loss",
                every_n_epochs=3,
                save_on_train_epoch_end=True,
                auto_insert_metric_name=True,
                dirpath="model_objects/",
                filename=f"{cfg.timestamp}"
                + "-{epoch}-{train_loss:.4f}-{train_acc:.4f}-{train_auroc:.4f}-{val_loss:.4f}-{val_acc:.4f}-{val_auroc:.4f}",
            )

        def forward(self, x):
            x = x.unsqueeze(1).expand(-1, 3, -1, -1)  # go from HxW → 3xHxW
            x = x.float() / 255
            # x = self.normalize(x)
            out = self.vit(x)

            return out

        def mixup_data(self, x, y):
            """
            Returns mixed inputs, pairs of targets, and lambda
            reference: mixup: Beyond Empirical Risk Minimization
            """
            lam = np.random.choice([0.2, 0.3, 0.4, 0.5])

            batch_size = x.size()[0]
            index = torch.randperm(batch_size)
            mixed_x = lam * x + (1 - lam) * x[index, :]
            y_a, y_b = y, y[index]
            return mixed_x, y_a, y_b, lam

        def mixup_loss_criterion(self, pred, y_a, y_b, lam):
            loss = lam * self.loss_criterion(pred, y_a) + (
                1 - lam
            ) * self.loss_criterion(pred, y_b)
            return loss

        def training_step(self, batch, batch_idx):
            x, y, y_ohe = batch

            # with prob 0.66 do mel spectogram mixup
            if np.random.choice([True, False, False]):
                y_pred = self(x)
                loss = self.loss_criterion(y_pred, y_ohe)
            else:
                mixed_x, y_a, y_b, lam = self.mixup_data(x, y_ohe)
                y_pred = self(mixed_x)
                loss = self.mixup_loss_criterion(y_pred, y_a, y_b, lam)

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
            x_val, y_val, y_val_ohe = batch
            y_pred = self(x_val)
            val_loss = self.loss_criterion(y_pred, y_val_ohe)
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
                optimizer, T_0=cfg.n_epochs, T_mult=1, eta_min=1e-7, last_epoch=-1
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
        fast_dev_run=True,
        max_epochs=cfg.n_epochs,
        accelerator=cfg.accelerator,
        enable_model_summary=False,
        callbacks=[model.checkpoint, TQDMProgressBar(refresh_rate=1)],
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
    trainer.save_checkpoint(f"model_objects/full_{cfg.run_tag}.ckpt")
    print("All done")
