from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from sklearn.metrics import mean_absolute_error, mean_squared_error


class RegressionModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        lr: float = 6e-6,
        weight_decay: float = 1e-5,
        optimizer: str = "adam",
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.net = net
        self.loss_fn = nn.SmoothL1Loss()

    def forward(self, y: torch.Tensor):
        return self.net(y)

    def step(self, batch: Any):
        y, y_true, _ = batch

        y_pred = self.forward(y)

        loss = self.loss_fn(y_pred, y_true)
        return loss, y_pred, y_true

    def training_step(self, batch: Any, batch_idx: int):
        loss, y_pred, y_true = self.step(batch)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        columns = ["loss", "MAE", "RMSE"]
        data = [loss, mae, rmse]

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/MAE", mae, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/RMSE", rmse, on_step=False, on_epoch=True, prog_bar=False)
        self.log_table(key="train", columns=columns, data=data)
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, y_pred, y_true = self.step(batch)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        columns = ["loss", "MAE", "RMSE"]
        data = [loss, mae, rmse]

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/MAE", mae, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/RMSE", rmse, on_step=False, on_epoch=True, prog_bar=False)
        self.log_table(key="val", columns=columns, data=data)
        return {"loss": loss, "preds": y_pred, "targets": y_true}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        pass

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        assert self.hparams.optimizer in ["adam", "adamw"]

        if self.hparams.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                params=self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(
                params=self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        return optimizer
