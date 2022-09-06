import torch
from torch import Tensor, nn

from config.config import NUM_DRUM, NUM_NOTE, NUM_PROGRAM, NUM_SPECIAL, NUM_TICK


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
    def __init__(self) -> None:
        super().__init__()
        self.special_limit = NUM_SPECIAL
        self.program_limit = self.special_limit + NUM_PROGRAM
        self.drum_limit = self.program_limit + NUM_DRUM
        self.note_limit = self.drum_limit + NUM_NOTE
        self.tick_limit = self.note_limit + NUM_TICK

    def forward(self, data: Tensor):
        return torch.cat(
            [
                data[:, : self.special_limit].sum(dim=1, keepdim=True),
                data[:, self.special_limit : self.program_limit].sum(dim=1, keepdim=True),
                data[:, self.program_limit : self.drum_limit].sum(dim=1, keepdim=True),
                data[:, self.drum_limit : self.note_limit].sum(dim=1, keepdim=True),
                data[:, self.note_limit : self.tick_limit].sum(dim=1, keepdim=True),
            ],
            dim=1,
        )
