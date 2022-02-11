"""The inferencing file.

Run this file with checkpoint specification for a single run of inference.

    Example:
        python inference.py best_checkpoint=\"checkpoints/best.ckpt\""""
import os
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
import numpy as np
from omegaconf import DictConfig
import torch
from torch import Tensor
import torch.nn.functional as F
from tqdm import tqdm

from data.datamodule import MusicDataModule
from data.utils import Modifier, events_to_midi, Converter
from model.model import MusicModel


def find_best_checkpoint(checkpoint_dir: str) -> str:
    """Searches the directory for checkpoint with minimum loss.

    Args:
        checkpoint_dir: Directory to search in (required).

    Returns:
        Checkpoint path with minimum loss.

    Examples:
        >>> best_ckpt = find_best_checkpoint(checkpoint_dir)"""
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
                min_filename = Path(dirpath, file)

    return to_absolute_path(str(min_filename))


def top_p_sampling(logits: Tensor, prob: float = 0.9) -> Tensor:
    """Top-p samples a 1D Tensor.

    Top-p sampling, also known as nucleus sampling, is performed by
    sampling only the top logits which sums to `p` or more.

    Args:
        logits: 1D Tensor of logits, before softmax (required).
        prob: Cutoff probability (required).

    Returns:
        Sampled prediction, a single indice between `1` and `d_input`.

    Shapes:
        logits: [d_input]
        prob: float
        output: [1]

    Examples:
        >>> pred = top_p_sampling(logits, prob)"""
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
    """The main function.

    Accepts a DictConfig and inferences the model accordingly.
    See `config/inference.yaml` for modifiable hyperparameters.

    Args:
        cfg: DictConfig housing all hyperparameters.

    Examples:
        >>> from hydra.utils import initialize, compose
        >>> with initialize(config_path=\"config\"):
        >>>     main(cfg=compose(config_name=\"inference\", overrides=...))"""

    # Initialize Modifier
    modifier = Modifier(
        num_special=cfg.data.num_special,
        num_program=cfg.data.num_program,
        num_note=cfg.data.num_note,
        num_velocity=cfg.data.num_velocity,
        num_time_num=cfg.data.num_time_num,
        note_shift=cfg.data.note_shift,
        velocity_scale=cfg.data.velocity_scale,
    )
    # Initialize DataModule
    datamodule = MusicDataModule(
        batch_size=1,
        data_dir=Path(to_absolute_path(Path(*cfg.data.data_dir))),
        text_dir=Path(to_absolute_path(Path(*cfg.data.text_dir))),
        process_dir=Path(to_absolute_path(Path(*cfg.data.process_dir))),
        num_workers=cfg.train.num_workers,
        data_len=cfg.model.data_len,
        augment=False,
        modifier=modifier,
    )

    # Get checkpoint path
    if cfg.best_checkpoint:
        # Get checkpoint with minimum loss
        checkpoint_dir = to_absolute_path("checkpoints")
        best_checkpoint = find_best_checkpoint(checkpoint_dir)
        print(f"Loading checkpoint {best_checkpoint}")
        model = MusicModel.load_from_checkpoint(
            find_best_checkpoint(checkpoint_dir))
    elif cfg.checkpoint_path:
        # Get checkpoint from specified path
        print(f"Loading checkpoint {cfg.checkpoint_path}")
        model = MusicModel.load_from_checkpoint(cfg.checkpoint_path)
    else:
        raise NotImplementedError("No checkpoint specified")

    # Initialize converter
    converter = Converter(token_type=np.uint8)
    # Setup datamodule
    datamodule.prepare_data()
    datamodule.setup()
    # Get dataloader
    dataloader = datamodule.test_dataloader()
    # Set model to eval mode
    model.eval()
    # Move model to GPU
    model.cuda()
    # Disable gradient calculation
    with torch.no_grad():
        batch = next(iter(dataloader))
        # Slice batch
        batch = batch[:cfg.model.data_len]
    with torch.no_grad():
        # Get batch
        batch = next(iter(dataloader))
        # Move batch to GPU
        batch = batch.cuda()
        # Slice batch
        data = batch[:, :cfg.model.data_len]
        # Convert data to tokens
        tokens = modifier.unflatten(data[0].detach().cpu().numpy())
        # Save tokens
        with open("input.txt", "w", encoding="utf-8") as file:
            for token in tokens:
                file.write(f"{token}\n")
        # Convert tokens to events
        events = converter.tokens_to_events(tokens)
        # Convert events to midi
        midi_file = events_to_midi(events)
        # Save midi
        midi_file.save(filename="input.mid")

        # Infer model
        for _ in tqdm(range(cfg.model.data_len)):
            logit = model(data)[0, :, -1]
            index = top_p_sampling(logit, prob=0.9)
            # Concate index to data
            data = torch.cat((data, index.unsqueeze(dim=0)), dim=1)
            # Slice data
            data = data[:, -cfg.model.data_len:]

        # Convert data to tokens
        tokens = modifier.unflatten(data[0].detach().cpu().numpy())
        # Save tokens
        with open("output.txt", "w", encoding="utf-8") as file:
            for token in tokens:
                file.write(f"{token}\n")
        # Convert tokens to events
        events = converter.tokens_to_events(tokens)
        # Convert events to midi
        midi_file = events_to_midi(events)
        # Save midi
        midi_file.save(filename="output.mid")


if __name__ == "__main__":
    main()
