import unittest

import torch
from hydra import compose, initialize

from model.embedding import Embedding


class TestEmbedding(unittest.TestCase):
    def setUp(self) -> None:
        with initialize(config_path="../config"):
            cfg = compose(config_name="config")
            self.cfg = cfg
            self.embedding = Embedding(
                d_model=cfg.model.d_model, num_token=cfg.model.num_token
            )

    def test_embedding(self):
        data = torch.zeros(8, self.cfg.model.data_len, dtype=torch.int64)
        output = self.embedding(data)
        self.assertEqual(
            output.size(), (8, self.cfg.model.data_len, self.cfg.model.d_model)
        )
