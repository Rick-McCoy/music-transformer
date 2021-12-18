import unittest

from hydra import initialize, compose
import torch

from model.transformer import Transformer


class TestTransformer(unittest.TestCase):
    def setUp(self) -> None:
        with initialize(config_path="../config"):
            cfg = compose(config_name="config")
            self.cfg = cfg
            self.transformer = Transformer(cfg)

    def test_transformer(self):
        data = torch.zeros(8, self.cfg.model.data_len, dtype=torch.int64)
        output = self.transformer(data)
        self.assertEqual(
            output.size(),
            (8, self.cfg.model.num_token, self.cfg.model.data_len))
