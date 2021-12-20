import os

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import torch
from tqdm import tqdm

from data.datamodule import MusicDataModule
from data.utils import Tokenizer, write_midi
from model.model import MusicModel


def find_best_checkpoint(checkpoint_dir: str) -> str:
    min_val_loss = 1e9
    min_filename = ""
    for dirpath, _, files in os.walk(checkpoint_dir):
        for file in files:
            filename = ".".join(file.split(".")[:-1])
            pairs = {}
            for key_value_pair in filename.split("-"):
                key, value = key_value_pair.split("=")
                pairs[key] = value

            if min_val_loss > float(pairs["val_loss"]):
                min_val_loss = float(pairs["val_loss"])
                min_filename = os.path.join(dirpath, file)

    return to_absolute_path(min_filename)


@hydra.main(config_path="config", config_name="inference")
def main(cfg: DictConfig = None) -> None:
    model = MusicModel(d_model=cfg.model.d_model,
                       data_len=cfg.model.data_len,
                       dropout=cfg.model.dropout,
                       ff=cfg.model.ff,
                       lr=cfg.train.lr,
                       nhead=cfg.model.nhead,
                       num_layers=cfg.model.num_layers,
                       num_token=cfg.model.num_token)
    datamodule = MusicDataModule(cfg)

    if cfg.best_checkpoint:
        checkpoint_dir = to_absolute_path("checkpoints")
        model.load_from_checkpoint(find_best_checkpoint(checkpoint_dir))
    elif cfg.checkpoint_path:
        model.load_from_checkpoint(cfg.checkpoint_path)
    else:
        raise NotImplementedError("No checkpoint specified")

    dataloader = datamodule.test_dataloader()
    batch = next(dataloader)[:1, :-1]
    for _ in tqdm(range(cfg.model.data_len)):
        pred = model(batch)[:, -1:]
        batch = torch.cat([batch, pred], dim=-1)[:, 1:]

    tokenizer = Tokenizer(cfg)
    tokens = batch[0].detach().cpu().numpy()
    note_list = tokenizer.tokens_to_notes(tokens)
    midi_file = write_midi(note_list)
    midi_file.save(filename="results.mid")


if __name__ == "__main__":
    main()
