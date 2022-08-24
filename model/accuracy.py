from torch import Tensor, nn


class SimpleAccuracy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ignore_index = 0

    def forward(self, logit: Tensor, target: Tensor) -> Tensor:
        pred = logit.argmax(dim=1)
        mask = target != self.ignore_index
        correct = pred.eq(target).mul(mask).sum()
        return correct / mask.sum()
