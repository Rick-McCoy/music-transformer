"""Unit test for `main.py`"""
import unittest

from hydra import initialize, compose

from main import main


class TestMain(unittest.TestCase):
    """Tester for `main.py`."""
    def test_fast_dev(self):
        """Tester for main.

        Simply runs the main function with fast_dev_run=True."""
        with initialize(config_path="../config"):
            cfg = compose(
                config_name="main",
                overrides=["train.fast_dev_run=True", "train.batch_size=2"])
            main(cfg)
