from torch import Tensor, nn
from torchmetrics import Accuracy


class SimpleAccuracy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.accuracy = Accuracy(top_k=1)

    def forward(self, logit: Tensor, target: Tensor) -> Tensor:
        return self.accuracy(logit, target)
