import unittest

import torch

from model.accuracy import SimpleAccuracy


class TestAccuracy(unittest.TestCase):
    def setUp(self) -> None:
        self.acc = SimpleAccuracy()

    def test_accuracy(self):
        logits = torch.arange(100, dtype=torch.float32).reshape(10, 10)
        target_1 = torch.full((10, ), 9)
        target_2 = torch.arange(10)
        self.assertAlmostEqual(self.acc(logits, target_1).numpy(), 1)
        self.assertAlmostEqual(self.acc(logits, target_2).numpy(), 0.1)
