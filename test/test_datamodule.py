import unittest

from torch import Tensor
from hydra import initialize, compose
import torch

from data.datamodule import SimpleDataModule


class TestDataModule(unittest.TestCase):
    def setUp(self) -> None:
        with initialize(config_path="../config"):
            cfg = compose(config_name="config")
            self.cfg = cfg
            self.module = SimpleDataModule(cfg)
        self.module.prepare_data()
        self.module.setup()

    def test_dataloader(self):
        data, label = next(iter(self.module.train_dataloader()))
        self.assertIsInstance(data, Tensor)
        self.assertEqual(data.dtype, torch.float32)
        self.assertEqual(
            data.size(),
            (self.cfg.train.batch_size, self.cfg.model.input_channels,
             self.cfg.model.h, self.cfg.model.w))
        self.assertIsInstance(label, Tensor)
        self.assertEqual(label.dtype, torch.int64)
        self.assertEqual(label.size(), (self.cfg.train.batch_size, ))
