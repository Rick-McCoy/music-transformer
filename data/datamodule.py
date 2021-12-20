import os
import random
from typing import List, Optional

from hydra.utils import to_absolute_path
import numpy as np
from numpy import ndarray
from numpy.lib.npyio import NpzFile
from omegaconf.dictconfig import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, random_split

from data.utils import prepare_data


class MusicDataset(Dataset):
    def __init__(self, cfg: DictConfig, key_list: List[str]) -> None:
        super().__init__()
        self.cfg = cfg
        self.length = cfg.model.data_len + 1
        self.key_list = key_list
        file_dir = os.path.join(*self.cfg.data.file_dir)
        file_path = to_absolute_path(os.path.join(file_dir, "midi.npz"))
        self.npz_file: NpzFile = np.load(file_path, mmap_mode="r")

    def __getitem__(self, index: int) -> ndarray:
        key = self.key_list[index]
        data = self.npz_file[key].astype(np.int64)
        orig_len = data.shape[0]
        if self.length > orig_len:
            return np.pad(data, (0, self.length - orig_len),
                          mode="constant",
                          constant_values=0)
        random_index = random.randint(0, orig_len - self.length)
        return data[random_index:random_index + self.length]

    def __len__(self):
        # return len(self.path_list)
        return len(self.key_list)


class MusicDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.train.batch_size
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def prepare_data(self) -> None:
        prepare_data(self.cfg)

    def setup(self, stage: Optional[str] = None) -> None:
        file_dir = os.path.join(*self.cfg.data.file_dir)
        file_path = to_absolute_path(os.path.join(file_dir, "midi.npz"))
        with np.load(file_path, mmap_mode="r") as file:
            key_list = list(file.keys())
        random.shuffle(key_list)
        val_len = test_len = int(len(key_list) * 0.1)
        train_len = len(key_list) - val_len - test_len
        full_key_list = key_list[:-test_len]
        test_key_list = key_list[-test_len:]
        if stage == "fit" or stage == "validate" or stage is None:
            full_dataset = MusicDataset(self.cfg, full_key_list)
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [train_len, val_len])
        if stage == "test" or stage == "predict" or stage is None:
            self.test_dataset = MusicDataset(self.cfg, test_key_list)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.cfg.train.num_workers,
                          pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.cfg.train.num_workers,
                          pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.cfg.train.num_workers,
                          pin_memory=True)
