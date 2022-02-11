"""Unit test for `model/pos_encoding.py`"""
import unittest

from hydra import initialize, compose
import torch

from model.pos_encoding import PositionalEncoding


class TestPositionalEncoding(unittest.TestCase):
    """Tester for `model/pos_encoding.py`."""
    def setUp(self) -> None:
        with initialize(config_path="../config"):
            cfg = compose(config_name="main")
            self.cfg = cfg
            self.pos_encoding = PositionalEncoding(d_model=cfg.model.d_model,
                                                   data_len=cfg.model.data_len,
                                                   dropout=cfg.model.dropout)

    def test_pos_encoding(self):
        """Tester for PositionalEncoding."""
        embedding = torch.zeros(8, self.cfg.model.data_len,
                                self.cfg.model.d_model)
        output = self.pos_encoding(embedding)
        self.assertEqual(output.size(),
                         (8, self.cfg.model.data_len, self.cfg.model.d_model))
