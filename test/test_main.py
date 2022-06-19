import unittest

from hydra import compose, initialize

from main import main


class TestMain(unittest.TestCase):
    def test_fast_dev(self):
        with initialize(config_path="../config", version_base=None):
            cfg = compose(
                config_name="config",
                overrides=["train.fast_dev_run=True", "train.batch_size=2"],
            )
            main(cfg)
