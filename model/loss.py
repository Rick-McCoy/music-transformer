from torch import Tensor, nn


class CrossEntropy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, logit: Tensor, target: Tensor) -> Tensor:
        return self.cross_entropy(logit, target)
