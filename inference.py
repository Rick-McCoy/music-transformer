from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor
from tqdm import tqdm

from config.config import CustomConfig
from data.datamodule import MusicDataModule
from data.utils import Tokenizer, write_midi
from model.model import MusicModel


def find_best_checkpoint(checkpoint_dir: Path) -> Path:
    min_val_loss = 1e9
    min_filename = Path("")
    for filename in checkpoint_dir.iterdir():
        # Filename is of the form: epoch={}-val_loss={}.ckpt
        val_loss = float(filename.stem.split("-")[1].split("=")[1])
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            min_filename = filename

    return min_filename


def top_p_sampling(logits: Tensor, prob: float = 0.9) -> Tensor:
    """
    Perform top-p sampling on the logits.
    Top-p sampling, like top-k sampling, is a method of sampling from a distribution
    that is defined by a list of probabilities.
    After sorting the probabilities in descending order, we calculate the cumulative sum
    of the probabilities.
    We then only consider up to the first probability that has a cumulative sum greater than
    or equal to `prob`, and ignore the rest.

    Args:
        logits: Tensor of logits (required).
        prob: Probability of sampling (default: 0.9).

    Returns:
        LongTensor of sampled indice.

    Shapes:
        - logits: (num_classes, )
        - output: (1, )
    """
    # Get probabilities from logits
    probs = F.softmax(logits, dim=-1)
    # Sort the probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    # Get the cumulative sum of the probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # Get the indice of the first cumulative probability that is greater than or equal to `prob`
    indice = torch.argmax((cumulative_probs >= prob).float(), dim=-1)
    # Zero out logits that are not sampled
    logits[sorted_indices[indice + 1 :]].fill_(-float("inf"))
    # Sample from the logits
    output = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    return output


@hydra.main(config_path="config", config_name="inference", version_base=None)
def main(cfg: DictConfig) -> None:
    custom_cfg = CustomConfig(cfg)
    datamodule = MusicDataModule(custom_cfg)

    if custom_cfg.best_checkpoint:
        best_checkpoint = find_best_checkpoint(custom_cfg.checkpoint_dir)
        print(f"Loading checkpoint {best_checkpoint}")
        model = MusicModel.load_from_checkpoint(str(best_checkpoint))
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
        batch: Tensor = next(iter(dataloader))
        tokens = batch[0].numpy()
        event_list = tokenizer.tokens_to_events(tokens)
        midi_file = write_midi(event_list)
        midi_file.save(filename="input.mid")
        with open("input_tokens.txt", mode="w", encoding="utf-8") as file:
            file.write(tokenizer.tokens_to_string(tokens))
        data = batch[:1].cuda()
        for _ in tqdm(range(custom_cfg.data_len)):
            output = model(data)[0, :, -1]
            pred = top_p_sampling(output, prob=0.9).unsqueeze(dim=0)
            data = torch.cat([data, pred], dim=-1)
        tokens = data[0].detach().cpu().numpy()
        event_list = tokenizer.tokens_to_events(tokens)
        midi_file = write_midi(event_list)
        midi_file.save(filename="results.mid")
        with open("output_tokens.txt", mode="w", encoding="utf-8") as file:
            file.write(tokenizer.tokens_to_string(tokens))


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
