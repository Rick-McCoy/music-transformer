import torch
from torch import Tensor, nn

from config.config import NUM_DRUM, NUM_NOTE, NUM_PROGRAM, NUM_SPECIAL, NUM_TICK


class SimplifyClass(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.special_limit = NUM_SPECIAL
        self.program_limit = self.special_limit + NUM_PROGRAM
        self.drum_limit = self.program_limit + NUM_DRUM
        self.note_limit = self.drum_limit + NUM_NOTE
        self.tick_limit = self.note_limit + NUM_TICK

    def forward(self, data: Tensor):
        data.masked_fill_(data < self.special_limit, 0)
        data.masked_fill_((self.special_limit <= data) & (data < self.program_limit), 1)
        data.masked_fill_((self.program_limit <= data) & (data < self.drum_limit), 2)
        data.masked_fill_((self.drum_limit <= data) & (data < self.note_limit), 3)
        data.masked_fill_((self.note_limit <= data) & (data < self.tick_limit), 4)
        return data


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
