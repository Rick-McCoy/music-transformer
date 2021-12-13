from omegaconf import DictConfig
import torch
from torch import Tensor


class SimpleAccuracy:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    def __call__(self, logit: Tensor, target: Tensor) -> Tensor:
        return torch.mean((torch.argmax(logit, dim=-1) == target).float())
