import unittest

import torch
from hydra import initialize, compose

from model.accuracy import SimpleAccuracy


class TestAccuracy(unittest.TestCase):
    def setUp(self) -> None:
        with initialize(config_path="../config"):
            cfg = compose(config_name="config")
            self.cfg = cfg
            self.acc = SimpleAccuracy(cfg)

    def test_accuracy(self):
        logits = torch.arange(100).reshape(10, 10)
        target_1 = torch.full((10, ), 9)
        target_2 = torch.arange(10)
        self.assertAlmostEqual(self.acc(logits, target_1), 1)
        self.assertAlmostEqual(self.acc(logits, target_2), 0.1)
