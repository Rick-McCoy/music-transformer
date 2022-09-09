import random
from pathlib import Path
from typing import List, Optional

import numpy as np
from mido import MidiFile
from mido.midifiles.meta import KeySignatureError
from numpy import ndarray
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, random_split
from tqdm import tqdm

from config.config import PAD, CustomConfig
from data.utils import augment_tokens, determine_on_notes, events_to_tokens, read_midi


class MusicDataset(Dataset):
    def __init__(self, cfg: CustomConfig, path_list: List[str], process_dir: Path) -> None:
        super().__init__()
        self.cfg = cfg
        self.length = cfg.data_len + 1
        self.path_list = path_list
        self.process_dir = process_dir

    def __getitem__(self, index: int) -> ndarray:
        path = Path(self.process_dir, self.path_list[index]).with_suffix(".npy")
        data: ndarray = np.load(path).astype(np.int64)
        data = augment_tokens(data)
        if self.length > data.shape[0]:
            on_notes = determine_on_notes(data)
            data = np.concatenate([on_notes, data], axis=0)[: self.length]
            pad_length = self.length - data.shape[0]
            return np.pad(data, (0, pad_length), mode="constant", constant_values=PAD)
        random_index = random.randint(0, data.shape[0] - self.length)
        prefix = data[:random_index]
        slice_data = data[random_index : random_index + self.length]
        on_notes = determine_on_notes(prefix)
        assert on_notes.shape[0] < self.length
        return np.concatenate([on_notes, slice_data])[: self.length]

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
        if not self.cfg.process_dir.is_dir():
            self.cfg.process_dir.mkdir(parents=True)
        filenames = []
        for path in tqdm(self.cfg.data_dir.glob("**/*.mid")):
            relative_path = path.relative_to(self.cfg.data_dir)
            filename = self.cfg.process_dir / relative_path.with_suffix(".npy")
            if filename.exists():
                continue
            filename.parent.mkdir(parents=True, exist_ok=True)
            try:
                midi_file = MidiFile(filename=path, clip=True)
            except (EOFError, KeySignatureError, IndexError) as exception:
                tqdm.write(f"{path} is invalid: {exception.__class__.__name__}")
                if delete_invalid_files:
                    path.unlink()
                continue
            event_list = read_midi(midi_file)
            tokens = events_to_tokens(event_list)
            np.save(filename, tokens.astype(np.int16))
            filenames.append(str(relative_path) + "\n")

        filenames.sort()
        text_path = self.cfg.file_dir / "midi.txt"
        if not text_path.exists():
            with open(text_path, mode="w", encoding="utf-8") as file:
                file.writelines(filenames)

    def setup(self, stage: Optional[str] = None) -> None:
        file_path = self.cfg.file_dir / "midi.txt"
        with open(file_path, mode="r", encoding="utf-8") as file:
            path_list = file.readlines()
        random.shuffle(path_list)
        split_length = int(len(path_list) * 0.1)
        train_len = len(path_list) - split_length * 2
        train_val_path_list = path_list[:-split_length]
        test_path_list = path_list[-split_length:]
        process_dir = self.cfg.process_dir
        if stage == "fit" or stage == "validate" or stage is None:
            train_val_dataset = MusicDataset(self.cfg, train_val_path_list, process_dir)
            self.train_dataset, self.val_dataset = random_split(
                train_val_dataset, [train_len, split_length]
            )
        if stage == "test" or stage == "predict" or stage is None:
            self.test_dataset = MusicDataset(self.cfg, test_path_list, process_dir)

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )
