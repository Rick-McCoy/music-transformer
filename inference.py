import os

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import torch
from torch import Tensor
import torch.nn.functional as F
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


@hydra.main(config_path="config", config_name="inference")
def main(cfg: DictConfig = None) -> None:
    datamodule = MusicDataModule(cfg)

    if cfg.best_checkpoint:
        checkpoint_dir = to_absolute_path("checkpoints")
        best_checkpoint = find_best_checkpoint(checkpoint_dir)
        print(f"Loading checkpoint {best_checkpoint}")
        model = MusicModel.load_from_checkpoint(
            find_best_checkpoint(checkpoint_dir))
    elif cfg.checkpoint_path:
        print(f"Loading checkpoint {cfg.checkpoint_path}")
        model = MusicModel.load_from_checkpoint(cfg.checkpoint_path)
    else:
        raise NotImplementedError("No checkpoint specified")

    tokenizer = Tokenizer(cfg)
    datamodule.prepare_data()
    datamodule.setup()
    dataloader = datamodule.test_dataloader()
    model.eval()
    model.freeze()
    with torch.no_grad():
        model.cuda()
        for batch in dataloader:
            batch = batch.cuda()
            data = batch[:1, :-1]
            tokens = data[0].detach().cpu().numpy()
            note_list = tokenizer.tokens_to_notes(tokens)
            midi_file = write_midi(note_list)
            midi_file.save(filename="input.mid")
            for _ in tqdm(range(cfg.model.data_len)):
                output = model(data)[0, :, -1]
                pred = top_p_sampling(output, prob=0.9).unsqueeze(dim=0)
                data = torch.cat([data, pred], dim=-1)[:, 1:]
            tokens = data[0].detach().cpu().numpy()
            note_list = tokenizer.tokens_to_notes(tokens)
            midi_file = write_midi(note_list)
            midi_file.save(filename="results.mid")
            break


if __name__ == "__main__":
    main()
