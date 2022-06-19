import os
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch import Tensor
from tqdm import tqdm

from config.config import CustomConfig
from data.datamodule import MusicDataModule
from data.utils import Tokenizer, write_midi
from model.model import MusicModel


def find_best_checkpoint(checkpoint_dir: Path) -> Path:
    min_val_loss = 1e9
    min_filename = ""
    for _, _, files in checkpoint_dir.iterdir():
        for file in files:
            # Filename: epoch={}-val_loss={}.ckpt
            file: Path
            if file.suffix == ".ckpt":
                filename = file.name
                val_loss = float(filename.split("-")[1].split("=")[1].split(".")[0])
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    min_filename = filename

    return checkpoint_dir / min_filename

def top_p_sampling(logits: Tensor, prob: float = 0.9) -> Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=0), dim=0)

    sorted_indices_to_remove = cumulative_probs > prob
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = -float("Inf")

    pred = torch.multinomial(F.softmax(logits, dim=0), num_samples=1)
    return pred


@hydra.main(config_path="config", config_name="inference", version_base=None)
def main(cfg: DictConfig = None) -> None:
    custom_cfg = CustomConfig(cfg)
    datamodule = MusicDataModule(custom_cfg)

    if custom_cfg.best_checkpoint:
        best_checkpoint = find_best_checkpoint(custom_cfg.checkpoint_dir)
        print(f"Loading checkpoint {best_checkpoint}")
        model = MusicModel.load_from_checkpoint(best_checkpoint)
    elif custom_cfg.checkpoint_path:
        print(f"Loading checkpoint {custom_cfg.checkpoint_path}")
        model = MusicModel.load_from_checkpoint(custom_cfg.checkpoint_path)
    else:
        raise NotImplementedError("No checkpoint specified")

    tokenizer = Tokenizer(custom_cfg)
    datamodule.prepare_data()
    datamodule.setup()
    dataloader = datamodule.test_dataloader()
    model.eval()
    model.freeze()
    with torch.no_grad():
        model.cuda()
        for batch in dataloader:
            batch: Tensor
            batch = batch.cuda()
            data = batch[:1, 1:]
            tokens = data[0].detach().cpu().numpy()
            event_list = tokenizer.tokens_to_events(tokens)
            midi_file = write_midi(event_list)
            midi_file.save(filename="input.mid")
            for _ in tqdm(range(custom_cfg.data_len)):
                output = model(data)[0, :, -1]
                pred = top_p_sampling(output, prob=0.9).unsqueeze(dim=0)
                data = torch.cat([data, pred], dim=-1)[:, 1:]
            tokens = data[0].detach().cpu().numpy()
            event_list = tokenizer.tokens_to_events(tokens)
            midi_file = write_midi(event_list)
            midi_file.save(filename="results.mid")
            break


if __name__ == "__main__":
    main()
