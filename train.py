import argparse
import math

import wandb
from hydra import compose, initialize

from main import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--effective_batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--d_model", type=int)
    parser.add_argument("--data_len", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--feed_forward", type=int)
    parser.add_argument("--nhead", type=int)
    parser.add_argument("--num_layers", type=int)
    args = parser.parse_args()
    wandb.init(config=args)
    config = wandb.config
    with initialize(config_path="config", version_base=None):
        segments = round(math.sqrt(config.num_layers))
        cfg = compose(
            config_name="config",
            overrides=[
                "train.auto_batch=True",
                "train.auto_lr=True",
                "train.gpus=1",
                "train.ignore_runtime_error=True",
                "train.limit_batches=0.1",
                f"train.effective_batch_size={config.effective_batch_size}",
                f"train.num_workers={config.num_workers}",
                f"model.d_model={config.d_model}",
                f"model.data_len={config.data_len}",
                f"model.dropout={config.dropout}",
                f"model.feed_forward={config.feed_forward}",
                f"model.nhead={config.nhead}",
                f"model.num_layers={config.num_layers}",
                f"model.segments={segments}",
            ],
        )
        main(cfg)
