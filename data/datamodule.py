import os
import random
from typing import List, Optional

from mido.midifiles.midifiles import MidiFile
import numpy as np
from omegaconf.dictconfig import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, random_split

from data.utils import read_midi, prepare_data


class MusicDataset(Dataset):
    def __init__(self, cfg: DictConfig, path_list: List[str]) -> None:
        super().__init__()
        self.cfg = cfg
        self.length = cfg.model.data_len + 1
        self.path_list = path_list

    def __getitem__(self, index: int):
        path = os.path.join(self.cfg.data.data_dir,
                            self.path_list[index].strip())
        ticks, programs, _, pitches, velocities = read_midi(
            MidiFile(filename=path, clip=True))

        assert len(ticks.shape) == 1
        assert len(programs.shape) == 1
        assert len(pitches.shape) == 1
        assert len(velocities.shape) == 1

        assert ticks.shape[0] == programs.shape[0]
        assert ticks.shape[0] == pitches.shape[0]
        assert ticks.shape[0] == velocities.shape[0]

        orig_len = ticks.shape[0]
        if orig_len >= self.length:
            random_index = random.randint(0, orig_len - self.length)
            return ticks[random_index:random_index + self.length], \
                   programs[random_index:random_index + self.length], \
                   pitches[random_index:random_index + self.length], \
                   velocities[random_index:random_index + self.length]

        return np.pad(ticks, (0, self.length - orig_len), mode="constant", constant_values=0), \
               np.pad(programs, (0, self.length - orig_len), mode="constant", constant_values=0), \
               np.pad(pitches, (0, self.length - orig_len), mode="constant", constant_values=0), \
               np.pad(velocities, (0, self.length - orig_len), mode="constant", constant_values=0)

    def __len__(self):
        return len(self.path_list)


class MusicDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.train.batch_size
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def prepare_data(self) -> None:
        prepare_data(self.cfg.data.data_dir, self.cfg.data.tar_dir)

    def setup(self, stage: Optional[str] = None) -> None:
        file_path = os.path.join(self.cfg.data.tar_dir, "midi.txt")
        with open(file_path, mode="r", encoding="utf-8") as file:
            path_list = file.readlines()
        random.shuffle(path_list)

        val_len = test_len = int(len(path_list) * 0.1)
        train_len = len(path_list) - val_len - test_len
        full_path_list = path_list[:-test_len]
        test_path_list = path_list[-test_len:]
        if stage == "fit" or stage == "validate" or stage is None:
            full_dataset = MusicDataset(self.cfg, full_path_list)
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [train_len, val_len])
        if stage == "test" or stage == "predict" or stage is None:
            self.test_dataset = MusicDataset(self.cfg, test_path_list)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.cfg.train.num_workers,
                          pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.cfg.train.num_workers,
                          pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.cfg.train.num_workers,
                          pin_memory=True)

    def teardown(self, stage: Optional[str] = None) -> None:
        return super().teardown(stage=stage)
