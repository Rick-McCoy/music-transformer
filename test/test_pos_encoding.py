import unittest

import torch
from hydra import compose, initialize

from model.pos_encoding import PositionalEncoding


class TestPositionalEncoding(unittest.TestCase):
    def setUp(self) -> None:
        with initialize(config_path="../config"):
            cfg = compose(config_name="config")
            self.cfg = cfg
            self.pos_encoding = PositionalEncoding(d_model=cfg.model.d_model,
                                                   data_len=cfg.model.data_len,
                                                   dropout=cfg.model.dropout)

    def test_pos_encoding(self):
        embedding = torch.zeros(8, self.cfg.model.data_len,
                                self.cfg.model.d_model)
        output = self.pos_encoding(embedding)
        self.assertEqual(output.size(),
                         (8, self.cfg.model.data_len, self.cfg.model.d_model))
