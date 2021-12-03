import numpy as np
from torch import Tensor
from omegaconf import DictConfig


class SimpleAccuracy:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    def __call__(self, logit: Tensor, target: Tensor) -> np.float32:
        return np.mean(
            np.argmax(logit.cpu().detach().numpy(), axis=-1) ==
            target.cpu().detach().numpy())
