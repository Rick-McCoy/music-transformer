import warnings

from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint_sequential

from model.embedding import Embedding
from model.pos_encoding import PositionalEncoding


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float,
        ff: int,
        nhead: int,
        data_len: int,
    ) -> None:
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        mask = nn.Transformer.generate_square_subsequent_mask(data_len)
        self.register_buffer("mask", mask)
        self.mask: Tensor

    def forward(self, data: Tensor):
        return self.layer(data, src_mask=self.mask)


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        data_len: int,
        dropout: float,
        ff: int,
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
        self.encoder = nn.Sequential(
            *[
                TransformerLayer(
                    d_model=d_model,
                    dropout=dropout,
                    ff=ff,
                    nhead=nhead,
                    data_len=data_len,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm((d_model,))
        self.linear = nn.Linear(
            in_features=d_model,
            out_features=num_tokens,
        )
        self.segments = segments

    def forward(self, data: Tensor) -> Tensor:
        embedded = self.embedding(data)
        encoded = self.pos_encoding(embedded)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            transformed = checkpoint_sequential(
                functions=self.encoder,
                segments=self.segments,
                input=encoded,
            )
        normalized = self.norm(transformed)
        projected: Tensor = self.linear(normalized)
        output = projected.permute([0, -1] + list(range(1, projected.ndim - 1)))
        return output
