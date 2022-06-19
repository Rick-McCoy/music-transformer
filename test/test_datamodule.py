import unittest

import torch
from hydra import compose, initialize
from torch import Tensor

from config.config import CustomConfig
from data.datamodule import MusicDataModule


class TestDataModule(unittest.TestCase):
    def setUp(self) -> None:
        with initialize(config_path="../config", version_base=None):
            cfg = compose(
                config_name="config",
                overrides=[
                    "train.fast_dev_run=True",
                    "train.batch_size=2",
                    "train.num_workers=1",
                ],
            )
            self.cfg = CustomConfig(cfg)
            self.module = MusicDataModule(self.cfg)
        self.module.prepare_data()
        self.module.setup()

    def test_dataloader(self):
        for dataloader in [
            self.module.train_dataloader(),
            self.module.val_dataloader(),
            self.module.test_dataloader(),
        ]:
            data = next(iter(dataloader))
            self.assertIsInstance(data, Tensor)
            self.assertEqual(data.dtype, torch.int64)
            self.assertEqual(
                data.size(),
                (
                    self.cfg.batch_size,
                    self.cfg.data_len + 1,
                ),
            )
            self.assertTrue(torch.all(data >= 0))
            self.assertTrue(torch.all(data < self.cfg.num_tokens))
