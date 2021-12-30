import argparse
import math

from hydra import initialize, compose
import wandb

from main import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    wandb.init(config=args)
    config = wandb.config
    with initialize(config_path="config"):
        segments = round(math.sqrt(config.num_layers))
        cfg = compose(config_name="config",
                      overrides=[
                          f"model.d_model={config.d_model}",
                          f"model.data_len={config.data_len}",
                          f"model.dropout={config.dropout}",
                          f"model.ff={config.ff}",
                          f"model.nhead={config.nhead}",
                          f"model.num_layers={config.num_layers}",
                          f"model.segments={segments}"
                      ])
        main(cfg)
