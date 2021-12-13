import unittest

from hydra import initialize, compose
import torch
from torch import Tensor

from data.datamodule import MusicDataModule


class TestDataModule(unittest.TestCase):
    def setUp(self) -> None:
        with initialize(config_path="../config"):
            cfg = compose(config_name="config")
            self.cfg = cfg
            self.module = MusicDataModule(cfg)
        self.module.prepare_data()
        self.module.setup()

    def test_dataloader(self):
        for dataloader in [
                self.module.train_dataloader(),
                self.module.val_dataloader(),
                self.module.test_dataloader()
        ]:
            ticks, programs, pitches, velocities = next(iter(dataloader))
            self.assertIsInstance(ticks, Tensor)
            self.assertEqual(ticks.dtype, torch.int64)
            self.assertEqual(ticks.size(), (
                self.cfg.train.batch_size,
                self.cfg.model.data_len + 1,
            ))
            self.assertIsInstance(programs, Tensor)
            self.assertEqual(programs.dtype, torch.int64)
            self.assertEqual(programs.size(), (
                self.cfg.train.batch_size,
                self.cfg.model.data_len + 1,
            ))
            self.assertIsInstance(pitches, Tensor)
            self.assertEqual(pitches.dtype, torch.int64)
            self.assertEqual(pitches.size(), (
                self.cfg.train.batch_size,
                self.cfg.model.data_len + 1,
            ))
            self.assertIsInstance(velocities, Tensor)
            self.assertEqual(velocities.dtype, torch.int64)
            self.assertEqual(velocities.size(), (
                self.cfg.train.batch_size,
                self.cfg.model.data_len + 1,
            ))
