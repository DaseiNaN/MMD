from typing import Any, List

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from src.utils.metrics import measure_performance


class ClassificationModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        lr: float = 6e-6,
        weight_decay: float = 1e-5,
        optimizer: str = "adamw",
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.net = net
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, y: torch.Tensor):
        return self.net(y)

    def step(self, batch: Any):
        y, _, y_true = batch

        y_pred = self.forward(y)
        y_pred = y_pred.max(dim=1, keepdim=True)[1]

        loss = self.loss_fn(y_pred, y_true)
        return loss, y_pred, y_true

    def training_step(self, batch: Any, batch_idx: int):
        loss, y_pred, y_true = self.step(batch)
        conf_matrix, [accuracy, precision, recall, f1_score] = measure_performance(y_pred, y_true)

        [[tp, fp], [fn, tn]] = conf_matrix
        columns = ["loss", "accuracy", "precision", "recall", "f1_score", "tp", "fp", "fn", "tn"]
        data = [loss, accuracy, precision, recall, f1_score, tp, fp, fn, tn]

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1_score", f1_score, on_step=False, on_epoch=True, prog_bar=True)
        self.log_table(key="train", columns=columns, data=data)
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, y_pred, y_true = self.step(batch)
        conf_matrix, [accuracy, precision, recall, f1_score] = measure_performance(y_pred, y_true)

        [[tp, fp], [fn, tn]] = conf_matrix
        columns = ["loss", "accuracy", "precision", "recall", "f1_score", "tp", "fp", "fn", "tn"]
        data = [loss, accuracy, precision, recall, f1_score, tp, fp, fn, tn]

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1_score", f1_score, on_step=False, on_epoch=True, prog_bar=True)
        self.log_table(key="val", columns=columns, data=data)
        return {"loss": loss, "preds": y_pred, "targets": y_true}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        pass

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        if self.hparams.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                params=self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        return optimizer
