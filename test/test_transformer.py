"""Unit test for `model/transformer.py`"""
import unittest

from hydra import initialize, compose
import torch

from model.transformer import Transformer


class TestTransformer(unittest.TestCase):
    """Tester for `model/transformer.py`."""
    def setUp(self) -> None:
        with initialize(config_path="../config"):
            cfg = compose(config_name="main")
            self.cfg = cfg
            self.num_token = cfg.data.num_special + cfg.data.num_program + \
                cfg.data.num_note + cfg.data.num_velocity + cfg.data.num_time_num + \
                    cfg.data.num_time_den
            self.transformer = Transformer(d_model=cfg.model.d_model,
                                           data_len=cfg.model.data_len,
                                           dropout=cfg.model.dropout,
                                           ff=cfg.model.ff,
                                           nhead=cfg.model.nhead,
                                           num_layer=cfg.model.num_layer,
                                           num_pos=cfg.model.num_pos,
                                           num_token=self.num_token,
                                           segments=cfg.model.segments)

    def test_transformer(self):
        """Tester for Transformer.

        Simply runs a sample input through the transformer.
        Checks for output shape."""
        data = torch.ones(8, self.cfg.model.data_len, dtype=torch.int64)
        pos = torch.ones(8,
                         self.cfg.model.data_len,
                         self.cfg.model.num_pos - 1,
                         dtype=torch.float)
        output = self.transformer(data, pos)
        self.assertEqual(output.size(),
                         (8, self.num_token, self.cfg.model.data_len))
        data = torch.ones(8, self.cfg.model.data_len * 2, dtype=torch.int64)
        pos = torch.ones(8,
                         self.cfg.model.data_len * 2,
                         self.cfg.model.num_pos - 1,
                         dtype=torch.float)
        output = self.transformer(data, pos)
        self.assertEqual(output.size(),
                         (8, self.num_token, self.cfg.model.data_len * 2))
