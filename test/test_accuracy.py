"""Unit test for `model/accuracy.py`."""
import unittest

import torch

from model.accuracy import SimpleAccuracy


class TestAccuracy(unittest.TestCase):
    """Tester for `SimpleAccuracy`."""
    def setUp(self) -> None:
        self.acc = SimpleAccuracy()

    def test_accuracy(self):
        """Testing function for `SimpleAccuracy`.

        Test case 1:
            - logit: arange(100).reshape(10, 10)
            - target: full(10, fill_value=9)
            - output: tensor(1)

        Test case 2:
            - logit: arange(100).reshape(10, 10)
            - target: full(10, fill_value=0)
            - output: tensor(0)

        Test case 3:
            - logit: arange(100).reshape(10, 10)
            - target: arange(10)
            - output: tensor(1 / 9)
            - Note that since index 0 is ignored, the accuracy is 1 / 9."""
        logits = torch.arange(100, dtype=torch.float32).reshape(10, 10)
        target_1 = torch.full((10, ), 9)
        target_2 = torch.zeros((10, ), dtype=torch.int64)
        target_3 = torch.arange(10, dtype=torch.int64)
        self.assertAlmostEqual(self.acc(logits, target_1).numpy(), 1)
        self.assertAlmostEqual(self.acc(logits, target_2).numpy(), 0)
        self.assertAlmostEqual(self.acc(logits, target_3).numpy(), 1 / 9)
