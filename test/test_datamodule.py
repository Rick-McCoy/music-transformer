"""Unit test for `data/datamodule.py`"""
from pathlib import Path
import unittest

from hydra import initialize, compose
import torch
from torch import Tensor

from data.datamodule import MusicDataModule
from data.utils import Modifier


class TestDataModule(unittest.TestCase):
    """Tester for `data/datamodule.py`."""
    def setUp(self) -> None:
        with initialize(config_path="../config"):
            cfg = compose(config_name="main",
                          overrides=[
                              "train.fast_dev_run=True", "train.batch_size=2",
                              "train.num_workers=1"
                          ])
            self.cfg = cfg
            # Initialize Modifier
            modifier = Modifier(num_special=cfg.data.num_special,
                                num_program=cfg.data.num_program,
                                num_note=cfg.data.num_note,
                                num_velocity=cfg.data.num_velocity,
                                num_time_num=cfg.data.num_time_num,
                                note_shift=cfg.data.note_shift,
                                velocity_scale=cfg.data.velocity_scale,
                                time_scale=cfg.data.time_scale)
            # Initialize DataModule
            self.module = MusicDataModule(
                batch_size=cfg.train.batch_size,
                data_dir=Path(*cfg.data.data_dir),
                filename_list=Path(*cfg.data.filename_list),
                process_dir=Path(*cfg.data.process_dir),
                num_workers=cfg.train.num_workers,
                data_len=cfg.model.data_len,
                augment=False,
                modifier=modifier)
        self.module.prepare_data()
        self.module.setup()

    def test_dataloader(self):
        """Tester for MusicDataModule.

        Checks if train, val, and test dataloaders are created correctly."""
        for dataloader in [
                self.module.train_dataloader(),
                self.module.val_dataloader(),
                self.module.test_dataloader()
        ]:
            data, position = next(iter(dataloader))
            self.assertIsInstance(data, Tensor)
            self.assertEqual(data.dtype, torch.int64)
            self.assertEqual(
                data.size(),
                (self.cfg.train.batch_size, self.cfg.model.data_len + 1))
            self.assertIsInstance(position, Tensor)
            self.assertEqual(position.dtype, torch.float32)
            self.assertEqual(
                position.size(),
                (self.cfg.train.batch_size, self.cfg.model.data_len + 1,
                 self.cfg.model.num_pos - 1))
