import argparse

from hydra import compose, initialize

import wandb
from main import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--effective_batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--feed_forward", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--data_len", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--d_model", type=int)
    parser.add_argument("--nhead", type=int)
    parser.add_argument("--num_workers", type=int)
    args = parser.parse_args()
    wandb.init(config=args)  # type: ignore
    with initialize(config_path="config", version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "train.auto_batch=False",
                "train.auto_lr=False",
                "train.batch_size=8",
                "train.gpus=1",
                "train.ignore_runtime_error=True",
                "train.limit_batches=0.1",
                f"train.effective_batch_size={args.effective_batch_size}",
                f"train.lr={args.learning_rate}",
                f"train.weight_decay={args.weight_decay}",
                f"train.num_workers={args.num_workers}",
                f"model.d_model={args.d_model}",
                f"model.data_len={args.data_len}",
                f"model.dropout={args.dropout}",
                f"model.feed_forward={args.feed_forward}",
                f"model.nhead={args.nhead}",
                f"model.num_layers={args.num_layers}",
            ],
        )
        main(cfg)
