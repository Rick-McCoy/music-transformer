"""Unit test for `model/loss.py`"""
import unittest

from hydra import initialize, compose
import numpy as np
import torch

from model.loss import SimpleLoss


class TestLoss(unittest.TestCase):
    """Tester for `model/loss.py`."""
    def setUp(self) -> None:
        with initialize(config_path="../config"):
            cfg = compose(config_name="main")
            self.cfg = cfg
            self.loss = SimpleLoss()

    def test_loss(self):
        """Tester for SimpleLoss.

        Test case 1:
            Identical logit values.
            Expected loss: log(num_token)

        Test case 2:
            -1e9 logit values, except for target indices.
            Expected loss: 0.0

        Test case 3:
            0 logit values, 0 indices.
            Expected loss: 0.0"""
        logit_1 = torch.ones(8, self.cfg.data.num_program,
                             self.cfg.model.data_len)
        target_1 = torch.ones(8, self.cfg.model.data_len, dtype=torch.int64)
        self.assertAlmostEqual(self.loss(logit_1, target_1).numpy(),
                               np.log(self.cfg.data.num_program),
                               places=4)
        logit_2 = torch.full(
            (8, self.cfg.data.num_program, self.cfg.model.data_len), -1e9)
        logit_2[torch.arange(8), torch.arange(8)] = 1e9
        target_2 = torch.arange(8, dtype=torch.int64).unsqueeze(dim=1).repeat(
            (1, self.cfg.model.data_len))
        self.assertAlmostEqual(self.loss(logit_2, target_2).numpy(),
                               0,
                               places=4)
        logit_3 = torch.zeros(8, self.cfg.data.num_program,
                              self.cfg.model.data_len)
        target_3 = torch.zeros(8, self.cfg.model.data_len, dtype=torch.int64)
        self.assertAlmostEqual(self.loss(logit_3, target_3).numpy(),
                               0,
                               places=4)
