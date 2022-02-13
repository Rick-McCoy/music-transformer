"""Implements Dataset and DataModule classes."""
from pathlib import Path
from typing import List, Optional

from hydra.utils import to_absolute_path
import numpy as np
from numpy import ndarray
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, random_split

from data.utils import Modifier, process_tokens


class MusicDataset(Dataset):
    """Music dataset class.

    Reads data from preprocessed files.

    Args:
        data_len: Length of data (required).
        augment: Whether to augment data (required).
        path_list: List of paths to preprocessed files (required).
        process_dir: Directory of preprocessed files (required).
        modifier: Modifier instance (required)."""
    def __init__(self, data_len: int, augment: bool, path_list: List[str],
                 process_dir: str, modifier: Modifier):
        super().__init__()
        self.pad = 0
        self.begin = 1
        self.end = 2
        self.length = data_len + 1
        self.augment = augment
        self.path_list = path_list
        self.process_dir = process_dir
        self.modifier = modifier

    def __getitem__(self, index: int) -> ndarray:
        """Get item corresponding to index.

        Extracts numpy array of tokens from preprocessed .npy file.
        Augments tokens if self.augment is True.
        Adds offset to tokens.
        Flattens tokens into 1D, removing zero velocity from NOTE_OFF events and
        removing zero delta time.
        Bookend tokens with START and END tokens.
        If the length of the tokens array is less than self.length, the tokens
        array is padded with PAD.
        PAD, START, END are defined as 0, 1, 2 respectively.
        Also returns temporal positions.

        Args:
            index: Index of item (required).

        Returns:
            Tuple of tokens, temporal positions

        Shape:
            tokens: (self.length,)
            positions: (self.length,)"""

        # Get filename of preprocessed file.
        filename = Path(self.process_dir,
                        self.path_list[index]).with_suffix(".npy")
        # Load preprocessed file.
        tokens = np.load(filename).astype(np.int64)
        # Augment tokens if self.augment is True.
        if self.augment:
            tokens = self.modifier.augment(tokens)
        # Flatten tokens and get temporal positions.
        tokens, positions = self.modifier.flatten(tokens)
        # Pad tokens and positions.
        tokens, positions = self.modifier.pad_or_slice(tokens, positions,
                                                       self.length)

        # Return tokens and temporal positions.
        return tokens, positions

    def __len__(self):
        """Get length of dataset."""
        return len(self.path_list)


class MusicDataModule(LightningDataModule):
    """Music DataModule class.

    Handles minutia of setting up datasets.

    Args:
        batch_size: Batch size (required).
        data_dir: Directory of MIDI files (required).
        text_dir: Directory of `midi.txt` (required).
        process_dir: Directory of preprocessed files (required).
        num_workers: Number of workers to use for multiprocessing (required).
        data_len: Length of data (required).
        augment: Whether to augment data (required).
        modifier: Modifier instance (required)."""
    def __init__(self, batch_size: int, data_dir: Path, text_dir: Path,
                 process_dir: Path, num_workers: int, data_len: int,
                 augment: bool, modifier: Modifier):
        super().__init__()
        self.batch_size = batch_size
        self.rng = np.random.default_rng()
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.text_dir = text_dir
        self.process_dir = process_dir
        self.data_len = data_len
        self.augment = augment
        self.modifier = modifier
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def prepare_data(self) -> None:
        process_tokens(self.data_dir, self.text_dir, self.process_dir)

    def setup(self, stage: Optional[str] = None) -> None:
        file_path = to_absolute_path(self.text_dir / "midi.txt")
        with open(file_path, mode="r", encoding="utf-8") as file:
            path_list = file.readlines()
        self.rng.shuffle(path_list)
        val_len = test_len = int(len(path_list) * 0.1)
        train_len = len(path_list) - val_len - test_len
        full_path_list = path_list[:-test_len]
        test_path_list = path_list[-test_len:]

        if stage == "fit" or stage == "validate" or stage is None:
            full_dataset = MusicDataset(data_len=self.data_len,
                                        augment=self.augment,
                                        path_list=full_path_list,
                                        process_dir=self.process_dir,
                                        modifier=self.modifier)
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [train_len, val_len])
        if stage == "test" or stage == "predict" or stage is None:
            self.test_dataset = MusicDataset(data_len=self.data_len,
                                             augment=self.augment,
                                             path_list=test_path_list,
                                             process_dir=self.process_dir,
                                             modifier=self.modifier)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True)
