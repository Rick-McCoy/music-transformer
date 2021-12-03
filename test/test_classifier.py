import unittest

import torch
from hydra import initialize, compose

from model.classifier import SimpleClassifier


class TestClassifier(unittest.TestCase):
    def setUp(self) -> None:
        with initialize(config_path="../config"):
            cfg = compose(config_name="config")
            self.cfg = cfg
            self.classifier = SimpleClassifier(cfg)

    def test_accuracy(self):
        data = torch.zeros(8, self.cfg.model.input_channels, self.cfg.model.h,
                           self.cfg.model.w)
        output = self.classifier(data)
        self.assertEqual(output.size(), (8, self.cfg.model.num_class))
