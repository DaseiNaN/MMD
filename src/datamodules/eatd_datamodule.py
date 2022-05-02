import copy
import os
import sys
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, Subset

from src.datamodules.components.eatd_dataset import EATDDataset


class BaseKFoldDataModule(LightningDataModule, ABC):
    @abstractmethod
    def setup_folds(self, num_folds: int) -> None:
        pass

    @abstractmethod
    def setup_fold_index(self, fold_index: int) -> None:
        pass


class EATDKFoldDataModule(BaseKFoldDataModule):
    def __init__(self, data_dir: str, data_type: str):
        super(EATDKFoldDataModule, self).__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.train_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.train_fold: Optional[Dataset] = None
        self.val_fold: Optional[Dataset] = None
        self.test_fold: Optional[Dataset] = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        self.eatd_dataset = EATDDataset(
            data_dir=self.hparams.data_dir, data_type=self.hparams.data_type
        )

    def setup_folds(self, num_folds: int) -> None:
        self.num_folds = num_folds
        y = self.eatd_dataset.y
        X = np.zeros(y.shape[0])
        self.splits = [split for split in StratifiedKFold(n_splits=num_folds).split(X, y)]

    def setup_fold_index(self, fold_index: int) -> None:
        train_indices, test_indices = self.splits[fold_index]
        val_indices = copy.deepcopy(test_indices)
        self.train_fold = Subset(self.eatd_dataset, train_indices)
        self.val_fold = Subset(self.eatd_dataset, val_indices)
        self.test_fold = Subset(self.eatd_dataset, test_indices)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_fold)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_fold)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_fold)

    def __post_init__(cls):
        super().__init__()


if __name__ == "__main__":
    data_dir = "/home/dasein/Projects/MMD/data/EATD-Feats"
    data_type = "audio"
    eatd_datamodule = EATDKFoldDataModule(data_dir, data_type)
    eatd_datamodule.setup_folds(2)
