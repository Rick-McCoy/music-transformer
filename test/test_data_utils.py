import os
import unittest

from hydra.compose import compose
from hydra.initialize import initialize
import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset

from data.utils import load_mnist, get_mnist


class TestDataUtils(unittest.TestCase):
    def setUp(self) -> None:
        with initialize(config_path="../config"):
            self.cfg = compose(config_name="config")

        load_mnist(".")
        (self.train, self.val), self.test = get_mnist(".")

    def test_load_mnist(self):
        self.assertTrue(os.path.isdir(os.path.join(".", "MNIST")))
        self.assertTrue(os.path.isdir(os.path.join(".", "MNIST", "raw")))
        self.assertTrue(
            os.path.isfile(
                os.path.join(".", "MNIST", "raw", "t10k-images-idx3-ubyte")))
        self.assertTrue(
            os.path.isfile(
                os.path.join(".", "MNIST", "raw", "t10k-labels-idx1-ubyte")))
        self.assertTrue(
            os.path.isfile(
                os.path.join(".", "MNIST", "raw", "train-images-idx3-ubyte")))
        self.assertTrue(
            os.path.isfile(
                os.path.join(".", "MNIST", "raw", "train-labels-idx1-ubyte")))

    def test_get_mnist(self):
        self.assertIsInstance(self.train, Dataset)
        self.assertIsInstance(self.val, Dataset)
        self.assertIsInstance(self.test, Dataset)

        data, label = next(iter(self.train))
        self.assertIsInstance(data, Tensor)
        self.assertIsInstance(label, int)
        self.assertEqual(data.size(), (self.cfg.model.input_channels,
                                       self.cfg.model.h, self.cfg.model.w))
        self.assertEqual(data.dtype, torch.float32)
