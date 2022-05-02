import os

import numpy as np
from torch.utils.data import Dataset


class EATDDataset(Dataset):
    def __init__(self, data_dir: str, data_type: str):
        super(EATDDataset, self).__init__()
        assert data_type in ["audio", "text", "fuse"]

        self.data_dir = data_dir
        self.data_type = data_type

        if self.data_type == "fuse":
            audio_feats_pkg = np.load(os.path.join(data_dir, "audio_feats.npz"))
            text_feats_pkg = np.load(os.path.join(data_dir, "text_feats.npz"))
            dep_idxs = audio_feats_pkg["dep_idxs"]
            self.audio_feats = audio_feats_pkg["feats"]
            self.text_feats = text_feats_pkg["feats"]
            self.y = np.zeros(shape=self.audio_feats.shape[0])
            self.y[dep_idxs] = 1
        else:
            feats_pkg = np.load(os.path.join(data_dir, f"{data_type}_feats.npz"))
            dep_idxs = feats_pkg["dep_idxs"]
            self.feats = feats_pkg["feats"]
            self.targets = feats_pkg["targets"]
            self.y = np.zeros(shape=self.feats.shape[0])
            self.y[dep_idxs] = 1

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index: int):
        if self.data_type == "fuse":
            return (
                self.audio_feats[index],
                self.text_feats[index],
                self.targets[index],
                self.y[index],
            )
        else:
            return self.feats[index], self.targets[index], self.y[index]


if __name__ == "__main__":
    data_dir = "/home/dasein/Projects/MMD/data/EATD-Feats"
    data_type = "audio"
    eatd_dataset = EATDDataset(data_dir, data_type)
