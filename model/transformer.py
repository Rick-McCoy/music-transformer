from typing import Tuple
from torch import nn, Tensor
import torch
from torch.nn import ModuleList
from omegaconf import DictConfig

from model.embedding import Embedding
from model.pos_encoding import PositionalEncoding


class Transformer(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.embedding = Embedding(cfg)
        self.pos_encoding = PositionalEncoding(cfg)
        self.common_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=cfg.model.d_model * 3,
                                       nhead=cfg.model.nhead,
                                       dim_feedforward=cfg.model.ff,
                                       dropout=cfg.model.dropout,
                                       batch_first=True),
            num_layers=cfg.model.num_layers,
            norm=nn.LayerNorm((cfg.model.d_model * 3, )))
        self.project = nn.Linear(in_features=cfg.model.d_model * 3,
                                 out_features=cfg.model.d_model)
        self.transformers = ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=cfg.model.d_model,
                                           nhead=cfg.model.nhead,
                                           dim_feedforward=cfg.model.ff,
                                           dropout=cfg.model.dropout,
                                           batch_first=True),
                num_layers=cfg.model.num_layers,
                norm=nn.LayerNorm((cfg.model.d_model, ))) for _ in range(4)
        ])
        mask = nn.Transformer.generate_square_subsequent_mask(
            cfg.model.data_len)
        self.register_buffer("mask", mask)
        self.mask: Tensor
        self.tick_linear = nn.Linear(in_features=cfg.model.d_model,
                                     out_features=cfg.model.num_tick)
        self.pitch_linear = nn.Linear(in_features=cfg.model.d_model,
                                      out_features=cfg.model.num_pitch)
        self.program_linear = nn.Linear(in_features=cfg.model.d_model,
                                        out_features=cfg.model.num_program)
        self.velocity_linear = nn.Linear(in_features=cfg.model.d_model,
                                         out_features=cfg.model.num_velocity)
        self.relu = nn.ReLU()

    def forward(self, tick: Tensor, pitch: Tensor, program: Tensor,
                velocity: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        embedded = self.embedding(pitch, program, velocity)
        encoded = self.pos_encoding(embedded, torch.cumsum(tick, dim=-1))
        projected = self.project(
            self.common_transformer(encoded, mask=self.mask))
        tick_out = self.tick_linear(
            self.relu(self.transformers[0](projected, mask=self.mask)))
        pitch_out = self.pitch_linear(
            self.relu(self.transformers[1](projected, mask=self.mask)))
        program_out = self.program_linear(
            self.relu(self.transformers[2](projected, mask=self.mask)))
        velocity_out = self.velocity_linear(
            self.relu(self.transformers[3](projected, mask=self.mask)))
        return tick_out, pitch_out, program_out, velocity_out
