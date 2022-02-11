"""Training file used in W&B sweeps.

Since sweeping hyperparamters requires some modifying
to the original training scheme, a separate file was created.
See internal ArgumentParser for modifiable hyperparameters.
Note that none of the parameters have default values.

Example:
    python train.py --d_model=128 --data_len=4096 ..."""
import argparse
import math

from hydra import initialize, compose
import wandb

from main import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int)
    parser.add_argument("--data_len", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--ff", type=int)
    parser.add_argument("--nhead", type=int)
    parser.add_argument("--num_layer", type=int)
    args = parser.parse_args()
    wandb.init(config=args)
    config = wandb.config
    with initialize(config_path="config"):
        segments = round(math.sqrt(config.num_layer))
        cfg = compose(
            config_name="main",
            overrides=[
                "train.auto_batch=True",
                "train.auto_lr=True",
                "train.effective_batch_size=64",
                "train.gpus=1",
                "train.limit_batches=0.1",
                f"model.d_model={config.d_model}",
                f"model.data_len={config.data_len}",
                f"model.dropout={config.dropout}",
                f"model.ff={config.ff}",
                f"model.nhead={config.nhead}",
                f"model.num_layer={config.num_layer}",
                f"model.segments={segments}",
            ],
        )
        main(cfg)
