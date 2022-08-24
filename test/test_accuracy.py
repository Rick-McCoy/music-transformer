import unittest

import torch

from model.accuracy import SimpleAccuracy


class TestAccuracy(unittest.TestCase):
    def setUp(self) -> None:
        self.acc = SimpleAccuracy()

    def test_accuracy(self):
        logits = torch.arange(200, dtype=torch.float32).reshape(10, 20)
        target_1 = torch.full((10,), 19, dtype=torch.long)
        target_2 = torch.arange(10, 20, dtype=torch.long)
        self.assertAlmostEqual(self.acc(logits, target_1).item(), 1)
        self.assertAlmostEqual(self.acc(logits, target_2).item(), 0.1)
