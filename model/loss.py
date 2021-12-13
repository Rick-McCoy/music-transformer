from omegaconf import DictConfig
from torch import nn, Tensor


class SimpleLoss(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, logit: Tensor, target: Tensor) -> Tensor:
        if target.dim() > 1:
            logit = logit.permute([0, -1] + list(range(1, logit.dim() - 1)))
        return self.cross_entropy(logit, target)
