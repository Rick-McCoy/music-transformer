import unittest

import torch
from hydra import compose, initialize

from model.transformer import Transformer


class TestTransformer(unittest.TestCase):
    def setUp(self) -> None:
        with initialize(config_path="../config"):
            cfg = compose(config_name="config")
            self.cfg = cfg
            self.transformer = Transformer(
                d_model=cfg.model.d_model,
                data_len=cfg.model.data_len,
                dropout=cfg.model.dropout,
                ff=cfg.model.ff,
                nhead=cfg.model.nhead,
                num_layers=cfg.model.num_layers,
                num_tokens=cfg.model.num_token,
                segments=cfg.model.segments,
            )

    def test_transformer(self):
        data = torch.zeros(8, self.cfg.model.data_len, dtype=torch.int64)
        output = self.transformer(data)
        self.assertEqual(output.size(), (8, self.cfg.model.num_token, self.cfg.model.data_len))
