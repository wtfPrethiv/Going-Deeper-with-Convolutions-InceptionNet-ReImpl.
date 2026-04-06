from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorflow.keras.utils import Progbar 

from models import InceptionNet, InceptionNetLoss
from utils import (
    AverageMeter,
    accuracy,
    get_logger,
    plot_training_curves,
    save_checkpoint,
    load_checkpoint,
    load_config,
)
from data.dataset import CIFAR10DataModule


class Trainer:
    def __init__(
        self,
        config_path: str | Path = "configs/configs.yaml",
        resume_from: str | Path | None = None,
        device: str = "auto",
    ) -> None:
        self.cfg = load_config(config_path)
        self.logger = get_logger("trainer", log_dir=self.cfg.paths.logs)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.logger.info(f"Using device: {self.device}")

        self.data_module = CIFAR10DataModule.from_config(self.cfg)
        self.train_loader, self.val_loader = self.data_module.get_loaders()
        self.logger.info(
            f"Train batches: {len(self.train_loader)} | Val batches: {len(self.val_loader)}"
        )

        self.model = InceptionNet(
            num_classes=self.cfg.model.num_classes,
            aux_logits=self.cfg.model.aux_logits,
            dropout=self.cfg.model.dropout,
        ).to(self.device)
        self.logger.info(f"Parameters: {self.model.count_params():,}")

        self.loss_fn = InceptionNetLoss(aux_weight=self.cfg.training.aux_loss_weight)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.cfg.training.lr,
            momentum=self.cfg.training.momentum,
            weight_decay=self.cfg.training.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.cfg.training.lr_step_size,
            gamma=self.cfg.training.lr_gamma,
        )

        self.history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
        }
        self.best_val_acc: float = 0.0
        self.start_epoch: int = 0

        if resume_from is not None:
            self._resume(resume_from)

    def fit(self) -> dict[str, list[float]]:
        total_epochs = self.cfg.training.epochs
        self.logger.info(f"Starting training — {total_epochs} epochs")

        for epoch in range(self.start_epoch, total_epochs):
            epoch_display = epoch + 1
            lr = self.scheduler.get_last_lr()[0]
            self.logger.info(f"\nEpoch {epoch_display}/{total_epochs} (lr={lr:.6f})")

            train_loss, train_acc = self._train_epoch()
            val_loss, val_acc = self._val_epoch()

            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            self.logger.info(
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc

            ckpt_path = Path(self.cfg.paths.checkpoints) / f"epoch_{epoch_display:03d}.pth"
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch_display,
                metrics={"val_acc": val_acc, "val_loss": val_loss},
                path=ckpt_path,
                is_best=is_best,
                best_path=self.cfg.paths.best_model,
            )

        self.logger.info(f"Training complete. Best val acc: {self.best_val_acc:.4f}")

        plot_training_curves(
            self.history,
            save_path=Path(self.cfg.paths.logs) / "training_curves.png",
        )
        return self.history

    def _train_epoch(self) -> tuple[float, float]:
        self.model.train()
        loss_meter = AverageMeter("train_loss")
        acc_meter = AverageMeter("train_acc")

        progbar = Progbar(target=len(self.train_loader))
        for batch_index, (batch_features, y_true) in enumerate(self.train_loader, 1):
            batch_features = batch_features.to(self.device, non_blocking=True)
            y_true = y_true.to(self.device, non_blocking=True)

            y_pred = self.model(batch_features)
            loss = self.loss_fn(y_pred, y_true)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            main_logits = y_pred[0] if isinstance(y_pred, tuple) else y_pred
            n = y_true.size(0)
            loss_meter.update(loss.item(), n)
            acc_meter.update(accuracy(main_logits, y_true), n)

            progbar.update(batch_index, values=[("loss", loss_meter.avg), ("acc", acc_meter.avg)])

        return loss_meter.avg, acc_meter.avg

    def _val_epoch(self) -> tuple[float, float]:
        self.model.eval()
        loss_meter = AverageMeter("val_loss")
        acc_meter = AverageMeter("val_acc")

        with torch.no_grad():
            progbar = Progbar(target=len(self.val_loader))
            for batch_index, (batch_features, y_true) in enumerate(self.val_loader, 1):
                batch_features = batch_features.to(self.device, non_blocking=True)
                y_true = y_true.to(self.device, non_blocking=True)

                outputs = self.model(batch_features)
                loss = self.loss_fn(outputs, y_true)

                n = y_true.size(0)
                loss_meter.update(loss.item(), n)
                acc_meter.update(accuracy(outputs, y_true), n)

                progbar.update(batch_index, values=[("val_loss", loss_meter.avg), ("val_acc", acc_meter.avg)])

        return loss_meter.avg, acc_meter.avg

    def _resume(self, path: str | Path) -> None:
        self.logger.info(f"Resuming from {path}")
        ckpt = load_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )
        self.start_epoch = ckpt["epoch"]
        self.best_val_acc = ckpt["metrics"].get("val_acc", 0.0)
        self.logger.info(f"Resumed from epoch {self.start_epoch}")