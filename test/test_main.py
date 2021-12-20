import unittest

from hydra import initialize, compose

from main import main


class TestMain(unittest.TestCase):
    def test_fast_dev(self):
        with initialize(config_path="../config"):
            cfg = compose(
                config_name="config",
                overrides=["train.fast_dev_run=True", "train.batch_size=2"])
            main(cfg)
