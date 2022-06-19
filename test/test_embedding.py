import unittest

import torch
from hydra import compose, initialize
from torch import Tensor

from config.config import CustomConfig
from model.embedding import Embedding


class TestEmbedding(unittest.TestCase):
    def setUp(self) -> None:
        with initialize(config_path="../config", version_base=None):
            cfg = compose(config_name="config")
            self.cfg = CustomConfig(cfg)
            self.embedding = Embedding(d_model=self.cfg.d_model, num_tokens=self.cfg.num_tokens)

    def test_embedding(self):
        data = torch.zeros(8, self.cfg.data_len, dtype=torch.int64)
        output: Tensor = self.embedding(data)
        self.assertEqual(output.size(), (8, self.cfg.data_len, self.cfg.d_model))
