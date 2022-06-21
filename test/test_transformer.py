import unittest

import torch
from hydra import compose, initialize

from config.config import CustomConfig
from model.transformer import Transformer


class TestTransformer(unittest.TestCase):
    def setUp(self) -> None:
        with initialize(config_path="../config", version_base=None):
            cfg = compose(config_name="config")
            self.cfg = CustomConfig(cfg)
            self.transformer = Transformer(
                d_model=self.cfg.d_model,
                data_len=self.cfg.data_len,
                dropout=self.cfg.dropout,
                feed_forward=self.cfg.feed_forward,
                nhead=self.cfg.nhead,
                num_layers=self.cfg.num_layers,
                num_tokens=self.cfg.num_tokens,
                segments=self.cfg.segments,
            )

    def test_transformer(self):
        data = torch.zeros(8, self.cfg.data_len, dtype=torch.int64)
        output = self.transformer(data)
        self.assertEqual(output.size(), (8, self.cfg.num_tokens, self.cfg.data_len))
