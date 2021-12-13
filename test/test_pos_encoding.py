import unittest

from hydra import initialize, compose
import torch

from model.pos_encoding import PositionalEncoding


class TestPositionalEncoding(unittest.TestCase):
    def setUp(self) -> None:
        with initialize(config_path="../config"):
            cfg = compose(config_name="config")
            self.cfg = cfg
            self.pos_encoding = PositionalEncoding(cfg)

    def test_pos_encoding(self):
        embedding = torch.zeros(8, self.cfg.model.data_len,
                                self.cfg.model.d_model * 3)
        time = torch.zeros(8, self.cfg.model.data_len, dtype=torch.int64)
        output = self.pos_encoding(embedding, time)
        self.assertEqual(
            output.size(),
            (8, self.cfg.model.data_len, self.cfg.model.d_model * 3))
