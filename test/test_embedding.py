"""Unit test for `model/embedding.py`"""
import unittest

from hydra import initialize, compose
import torch

from model.embedding import Embedding


class TestEmbedding(unittest.TestCase):
    """Tester for `model/embedding.py`."""
    def setUp(self) -> None:
        with initialize(config_path="../config"):
            cfg = compose(config_name="main")
            self.cfg = cfg
            self.embedding = Embedding(d_model=cfg.model.d_model,
                                       num_token=cfg.data.num_program)

    def test_embedding(self):
        """Tester for Embedding.

        Tests if embedding returns correct shape."""
        embedding = self.embedding(torch.tensor([0, 1, 2, 3]))
        self.assertEqual(embedding.size(), (4, self.cfg.model.d_model))
