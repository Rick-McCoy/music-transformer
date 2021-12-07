import unittest

import torch
from hydra import initialize, compose

from model.embedding import Embedding


class TestEmbedding(unittest.TestCase):
    def setUp(self) -> None:
        with initialize(config_path="../config"):
            cfg = compose(config_name="config")
            self.cfg = cfg
            self.embedding = Embedding(cfg)

    def test_embedding(self):
        pitch = torch.zeros(8, self.cfg.model.data_len, dtype=torch.int64)
        program = torch.zeros(8, self.cfg.model.data_len, dtype=torch.int64)
        velocity = torch.zeros(8, self.cfg.model.data_len, dtype=torch.int64)
        output = self.embedding(pitch, program, velocity)
        self.assertEqual(
            output.size(),
            (8, self.cfg.model.data_len, self.cfg.model.d_model * 3))
