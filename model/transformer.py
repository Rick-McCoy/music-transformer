from omegaconf import DictConfig
from torch import nn, Tensor

from model.embedding import Embedding
from model.pos_encoding import PositionalEncoding


class Transformer(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.embedding = Embedding(cfg)
        self.pos_encoding = PositionalEncoding(cfg)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=cfg.model.d_model,
                                       nhead=cfg.model.nhead,
                                       dim_feedforward=cfg.model.ff,
                                       dropout=cfg.model.dropout,
                                       batch_first=True),
            num_layers=cfg.model.num_layers,
            norm=nn.LayerNorm((cfg.model.d_model, )))
        mask = nn.Transformer.generate_square_subsequent_mask(
            cfg.model.data_len)
        self.register_buffer("mask", mask)
        self.mask: Tensor
        self.linear = nn.Linear(in_features=cfg.model.d_model,
                                out_features=cfg.model.num_token)

    def forward(self, data: Tensor) -> Tensor:
        embedded = self.embedding(data)
        encoded = self.pos_encoding(embedded)
        transformed = self.transformer(encoded, mask=self.mask)
        projected = self.linear(transformed)
        output = projected.permute([0, -1] +
                                   list(range(1, projected.ndim - 1)))
        return output
