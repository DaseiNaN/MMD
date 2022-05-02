from typing import Any, List, Type

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning.core.lightning import LightningModule
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.utils.metrics import measure_performance


class EnsembleVotingModule(LightningModule):
    def __init__(
        self, model_cls: Type[LightningModule], checkpoint_paths: List[str], model_type: str
    ) -> None:
        super().__init__()
        assert model_type in ["classification", "regression", "fuse"]

        # Create `num_folds` models with their associated fold weights
        self.models = torch.nn.ModuleList(
            [model_cls.load_from_checkpoint(p) for p in checkpoint_paths]
        )
        self.model_type = model_type

        if self.model_type == "classification":
            self.loss_fn = nn.CrossEntropyLoss()
        elif self.model_type == "regression":
            self.loss_fn = nn.SmoothL1Loss()
        else:
            # TODO: fuse loss function
            self.loss_fn = None

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        # Compute the averaged predictions over the `num_folds` models.
        if self.model_type == "classification":
            y, _, y_true = batch
            y_pred = torch.stack([m(y) for m in self.models]).mean(0)
            y_pred = y_pred.max(dim=1, keepdim=True)[1]

            loss = self.loss_fn(y_pred, y_true)
            conf_matrix, [accuracy, precision, recall, f1_score] = measure_performance(
                y_pred, y_true
            )

            [[tp, fp], [fn, tn]] = conf_matrix
            columns = [
                "loss",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "tp",
                "fp",
                "fn",
                "tn",
            ]
            data = [loss, accuracy, precision, recall, f1_score, tp, fp, fn, tn]

            self.log("k-fold-test/loss", loss, on_step=False, on_epoch=True)
            self.log("k-fold-test/accuracy", accuracy, on_step=False, on_epoch=True)
            self.log_table(
                key="k-fold-test", columns=columns, data=data, on_step=False, on_epoch=True
            )
        elif self.mode == "regression":
            y, y_true, _ = batch
            y_pred = torch.stack([m(y) for m in self.models]).mean(0)
            loss = self.loss_fn(y_pred, y_true)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            columns = ["loss", "MAE", "RMSE"]
            data = [loss, mae, rmse]

            self.log("k-fold-test/loss", loss, on_step=False, on_epoch=True)
            self.log("k-fold-test/MAE", mae, on_step=False, on_epoch=True)
            self.log("k-fold-test/RMSE", rmse, on_step=False, on_epoch=True)
            self.log_table(
                key="k-fold-test", columns=columns, data=data, on_step=False, on_epoch=True
            )
        elif self.model_type == "fuse":
            # TODO: fuse test step
            pass
