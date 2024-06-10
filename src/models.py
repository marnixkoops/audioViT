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
        self.loss_function = FocalLossBCE()

        self.accuracy_top1 = torchmetrics.Accuracy(
            task="multiclass", num_classes=cfg.num_classes, top_k=1
        )
        self.accuracy_top2 = torchmetrics.Accuracy(
            task="multiclass", num_classes=cfg.num_classes, top_k=2
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
        self.lrap = torchmetrics.classification.MultilabelRankingAveragePrecision(
            num_labels=cfg.num_classes
        )

    def melspec_mixup(
        self,
        x: torch.tensor,
        y: torch.tensor,
        secondary_target_weight: float = None,
    ):
        mix_lambda = np.random.choice(np.arange(start=0.2, stop=0.55, step=0.05))

        if secondary_target_weight is None:
            secondary_target_weight = mix_lambda * 1

        batch_size = x.size()[0]
        batch_index = torch.randperm(batch_size)
        mixed_x = mix_lambda * x + (1 - mix_lambda) * x[batch_index, :]
        mixed_y = y + (y[batch_index] * secondary_target_weight)
        return mixed_x, mixed_y

    def forward(self, x):
        x = x.unsqueeze(1).expand(-1, 3, -1, -1)  # go from HxW → 3xHxW
        x = x.float() / 255
        x = self.imagenet_normalize(x)
        out = self.vit(x)

        return out

    def training_step(self, batch, batch_idx):
        x, y, y_binary, y_primary = batch

        if cfg.melspec_mixup:
            if np.random.random() < cfg.melspec_mixup_prob:
                x, y = self.melspec_mixup(x, y, secondary_target_weight=1.0)

        y_pred = self(x)
        loss = self.loss_function(y_pred, y)

        y_pred = y_pred.softmax(dim=1)
        train_accuracy_top1 = self.accuracy_top1(y_pred, y_primary)
        train_accuracy_top2 = self.accuracy_top2(y_pred, y_primary)
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
            "train_acc_1",
            train_accuracy_top1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_acc_2",
            train_accuracy_top2,
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

        y_pred = y_pred.softmax(dim=1)
        val_accuracy_top1 = self.accuracy_top1(y_pred, y_primary_val)
        val_accuracy_top2 = self.accuracy_top2(y_pred, y_primary_val)
        val_f1_weighted = self.f1_weighted(y_pred, y_binary_val)
        val_f1_macro = self.f1_macro(y_pred, y_binary_val)
        val_auroc = self.auroc(y_pred, y_binary_val)
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
            "val_acc_1",
            val_accuracy_top1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_acc_2",
            val_accuracy_top2,
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
