import random
from pathlib import Path
from typing import List, Optional

import numpy as np
from numpy import ndarray
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, random_split

from config.config import CustomConfig
from data.utils import prepare_data


class MusicDataset(Dataset):
    def __init__(self, cfg: CustomConfig, path_list: List[str], process_dir: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.length = cfg.data_len + 1
        self.path_list = path_list
        self.process_dir = process_dir

    def __getitem__(self, index: int) -> ndarray:
        path = Path(self.process_dir, self.path_list[index]).with_suffix(".npy")
        data: ndarray = np.load(path).astype(np.int64)
        orig_len = data.shape[0]
        if self.length > orig_len:
            return np.pad(data, (0, self.length - orig_len), mode="constant", constant_values=0)
        random_index = random.randint(0, orig_len - self.length)
        return data[random_index : random_index + self.length]

    def __len__(self):
        return len(self.path_list)


class MusicDataModule(LightningDataModule):
    def __init__(self, cfg: CustomConfig):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def prepare_data(self, delete_invalid_files: bool = False) -> None:
        prepare_data(self.cfg, delete_invalid_files)

    def setup(self, stage: Optional[str] = None) -> None:
        file_path = self.cfg.file_dir / "midi.txt"
        with open(file_path, mode="r", encoding="utf-8") as file:
            path_list = file.readlines()
        random.shuffle(path_list)
        val_len = test_len = int(len(path_list) * 0.1)
        train_len = len(path_list) - val_len - test_len
        full_path_list = path_list[:-test_len]
        test_path_list = path_list[-test_len:]
        process_dir = self.cfg.process_dir
        if stage == "fit" or stage == "validate" or stage is None:
            full_dataset = MusicDataset(self.cfg, full_path_list, process_dir)
            self.train_dataset, self.val_dataset = random_split(full_dataset, [train_len, val_len])
        if stage == "test" or stage == "predict" or stage is None:
            self.test_dataset = MusicDataset(self.cfg, test_path_list, process_dir)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )
