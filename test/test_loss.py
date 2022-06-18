import unittest

import numpy as np
import torch
from hydra import compose, initialize

from config.config import CustomConfig
from model.loss import CrossEntropy


class TestLoss(unittest.TestCase):
    def setUp(self) -> None:
        with initialize(config_path="../config"):
            cfg = compose(config_name="config")
            self.cfg = CustomConfig(cfg)
            self.loss = CrossEntropy()

    def test_loss(self):
        logit_1 = torch.ones(8, self.cfg.num_tokens, self.cfg.data_len)
        target_1 = torch.ones(8, self.cfg.data_len, dtype=torch.int64)
        self.assertAlmostEqual(
            self.loss(logit_1, target_1).numpy(),
            np.log(self.cfg.num_tokens),
            places=4,
        )
        logit_2 = torch.full((8, self.cfg.num_tokens, self.cfg.data_len), -1e9)
        logit_2[torch.arange(8), torch.arange(8)] = 1e9
        target_2 = (
            torch.arange(8, dtype=torch.int64)
            .unsqueeze(dim=1)
            .repeat((1, self.cfg.data_len))
        )
        self.assertAlmostEqual(self.loss(logit_2, target_2).numpy(), 0, places=4)
