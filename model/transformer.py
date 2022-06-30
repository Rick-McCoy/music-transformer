import copy

import torch
from torch import Tensor, nn

from model.embedding import Embedding
from model.pos_encoding import PositionalEncoding


class CheckpointEncoder(nn.Module):
    def __init__(self, layer: nn.Module, num_layers: int, norm: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, data: Tensor, mask: Tensor) -> Tensor:
        for layer in self.layers:
            data = torch.utils.checkpoint.checkpoint(layer, data, mask)
        return data

class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        data_len: int,
        dropout: float,
        feed_forward: int,
        nhead: int,
        num_layers: int,
        num_tokens: int,
        segments: int,
    ) -> None:
        super().__init__()
        self.embedding = Embedding(
            d_model=d_model,
            num_tokens=num_tokens,
        )
        self.pos_encoding = PositionalEncoding(
            d_model=d_model,
            data_len=data_len,
            dropout=dropout,
        )
        mask = nn.Transformer.generate_square_subsequent_mask(data_len)
        self.register_buffer("mask", mask)
        self.mask: Tensor
        self.encoder = CheckpointEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=feed_forward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm((d_model,)),
        )
        self.linear = nn.Linear(
            in_features=d_model,
            out_features=num_tokens,
        )
        self.segments = segments

    def forward(self, data: Tensor) -> Tensor:
        embedded = self.embedding(data)
        encoded = self.pos_encoding(embedded)
        normalized = self.encoder(encoded, mask=self.mask)
        projected: Tensor = self.linear(normalized)
        output = projected.permute([0, -1] + list(range(1, projected.ndim - 1)))
        return output
