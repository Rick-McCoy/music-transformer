import torch
from torch import Tensor, nn

from config.config import (
    DRUM_LIMIT,
    NOTE_LIMIT,
    NUM_DRUM,
    NUM_NOTE,
    NUM_PROGRAM,
    NUM_SPECIAL,
    NUM_TICK,
    PROGRAM_LIMIT,
    SPECIAL_LIMIT,
    TICK_LIMIT,
)


class SimplifyClass(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.simple_class = torch.LongTensor(
            [0] * NUM_SPECIAL
            + [1] * NUM_PROGRAM
            + [2] * NUM_DRUM
            + [3] * NUM_NOTE
            + [4] * NUM_TICK
        )

    def forward(self, data: Tensor):
        return self.simple_class[data]


class SimplifyScore(nn.Module):
    def forward(self, data: Tensor):
        return torch.cat(
            [
                data[:, :SPECIAL_LIMIT].sum(dim=1, keepdim=True),
                data[:, SPECIAL_LIMIT:PROGRAM_LIMIT].sum(dim=1, keepdim=True),
                data[:, PROGRAM_LIMIT:DRUM_LIMIT].sum(dim=1, keepdim=True),
                data[:, DRUM_LIMIT:NOTE_LIMIT].sum(dim=1, keepdim=True),
                data[:, NOTE_LIMIT:TICK_LIMIT].sum(dim=1, keepdim=True),
            ],
            dim=1,
        )
