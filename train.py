import argparse

import wandb
from hydra import initialize, compose

from main import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    wandb.init(config=args)
    config = wandb.config
    with initialize(config_path="config"):
        cfg = compose(config_name="config",
                      overrides=[
                          f"model.d_model={config.d_model}",
                          f"model.data_len={config.data_len}",
                          f"model.dropout={config.dropout}",
                          f"model.ff={config.ff}",
                          f"model.nhead={config.nhead}"
                      ])
        main(cfg)
