import torchmetrics
from torch import Tensor, nn


class SimpleAccuracy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.accuracy = torchmetrics.Accuracy(top_k=1)

    def forward(self, logit: Tensor, target: Tensor) -> Tensor:
        return self.accuracy(logit, target)
