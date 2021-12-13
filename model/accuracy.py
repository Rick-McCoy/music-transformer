from omegaconf import DictConfig
from torch import nn, Tensor
from torchmetrics import Accuracy


class SimpleAccuracy(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.accuracy = Accuracy(top_k=1)

    def forward(self, logit: Tensor, target: Tensor) -> Tensor:
        return self.accuracy(logit, target)
