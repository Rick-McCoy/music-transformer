import math
import unittest

import torch
from hydra import compose, initialize
from torch import Tensor

from config.config import CustomConfig
from model.pos_encoding import PositionalEncoding


class TestPositionalEncoding(unittest.TestCase):
    def setUp(self) -> None:
        with initialize(config_path="../config", version_base=None):
            cfg = compose(config_name="config")
            self.cfg = CustomConfig(cfg)
            self.pos_encoding = PositionalEncoding(
                d_model=self.cfg.d_model,
                data_len=self.cfg.data_len,
                dropout=self.cfg.dropout,
            )
            self.pos_encoding.eval()

    def test_pos_encoding(self):
        embedding = torch.zeros(8, self.cfg.data_len, self.cfg.d_model)
        output: Tensor = self.pos_encoding(embedding)
        self.assertEqual(output.size(), (8, self.cfg.data_len, self.cfg.d_model))
        position = torch.arange(0, self.cfg.data_len).reshape(-1, 1) * torch.exp(
            torch.arange(0, self.cfg.d_model, 2) * (-math.log(10000.0) / self.cfg.d_model)
        )
        self.assertAlmostEqual(
            torch.sum(torch.abs(output[0, :, 0::2] - torch.sin(position))).item(),
            0.0,
            places=4,
        )
        self.assertAlmostEqual(
            torch.sum(torch.abs(output[0, :, 1::2] - torch.cos(position))).item(),
            0.0,
            places=4,
        )
