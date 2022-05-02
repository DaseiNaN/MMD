from typing import Any, List, Type

import torch
import torch.nn as nn
from pytorch_lightning.core.lightning import LightningModule

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
            self.loss_fn = nn.L1Loss()
        else:
            # TODO: fuse loss function
            self.loss_fn = None

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        # Compute the averaged predictions over the `num_folds` models.
        if self.model_type == "classification":
            y_pred = torch.stack([m(batch[0]) for m in self.models]).mean(0)
            y_pred = y_pred.max(dim=1, keepdim=True)[1]

            loss = self.loss_fn(y_pred, batch[1])
            conf_matrix, [accuracy, precision, recall, f1_score] = measure_performance(
                y_pred, batch[1]
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

            self.log("k-fold-test/loss", loss)
            self.log("k-fold-test/accuracy", accuracy)
            self.log_table(key="k-fold-test", columns=columns, data=data)
        elif self.mode == "regression":
            # TODO: regression test step
            pass
        elif self.model_type == "fuse":
            # TODO: fuse test step
            pass
