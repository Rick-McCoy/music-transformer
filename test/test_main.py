import unittest

from hydra import compose, initialize

from main import main


class TestMain(unittest.TestCase):
    def test_fast_dev(self):
        with initialize(config_path="../config", version_base=None):
            cfg = compose(
                config_name="config",
                overrides=[
                    "train.auto_batch=False",
                    "train.auto_lr=False",
                    "train.fast_dev_run=True",
                    "train.batch_size=2",
                    "train.num_workers=4",
                    "train.wandb=False",
                ],
            )
            main(cfg)
