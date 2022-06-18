import unittest

import torch
from hydra import compose, initialize
from torch import Tensor

from data.datamodule import MusicDataModule


class TestDataModule(unittest.TestCase):
    def setUp(self) -> None:
        with initialize(config_path="../config"):
            cfg = compose(
                config_name="config",
                overrides=[
                    "train.fast_dev_run=True",
                    "train.batch_size=2",
                    "train.num_workers=1",
                ],
            )
            self.cfg = cfg
            self.module = MusicDataModule(cfg)
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
                    self.cfg.train.batch_size,
                    self.cfg.model.data_len + 1,
                ),
            )
