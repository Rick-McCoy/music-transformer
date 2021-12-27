from copy import deepcopy
from torch import nn, Tensor

from model.embedding import Embedding
from model.pos_encoding import PositionalEncoding


class Transformer(nn.Module):
    def __init__(self, d_model: int, data_len: int, dropout: float, ff: int,
                 nhead: int, num_layers: int, num_token: int) -> None:
        super().__init__()
        self.embedding = Embedding(d_model=d_model, num_token=num_token)
        self.pos_encoding = PositionalEncoding(d_model=d_model,
                                               data_len=data_len,
                                               dropout=dropout)
        layer = nn.TransformerEncoderLayer(d_model=d_model,
                                           nhead=nhead,
                                           dim_feedforward=ff,
                                           dropout=dropout,
                                           batch_first=True,
                                           norm_first=True)
        self.layers = [deepcopy(layer) for _ in range(num_layers)]
        self.norm = nn.LayerNorm((d_model, ))
        mask = nn.Transformer.generate_square_subsequent_mask(data_len)
        self.register_buffer("mask", mask)
        self.mask: Tensor
        self.linear = nn.Linear(in_features=d_model, out_features=num_token)

    def forward(self, data: Tensor) -> Tensor:
        embedded = self.embedding(data)
        encoded = self.pos_encoding(embedded)
        for layer in self.layers:
            encoded = layer(encoded, src_mask=self.mask)
        transformed = self.norm(encoded)
        projected = self.linear(transformed)
        output = projected.permute([0, -1] +
                                   list(range(1, projected.ndim - 1)))
        return output
