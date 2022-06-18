import math

import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \\text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \\text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \\text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model: int, data_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, data_len, dtype=torch.float).reshape(1, -1, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        term = position * div_term
        positional_encoding = torch.flatten(
            torch.stack([torch.sin(term), torch.cos(term)], dim=-1), start_dim=-2
        )
        self.positional_encoding: Tensor = None
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, embedding: Tensor):
        r"""Inputs of forward function
        Args:
            embedding: the sequence fed to the positional encoder model (required).
        Shape:
            embedding: [batch size, sequence length, model dim]
            output: [batch size, sequence length, model dim]
        Examples:
            >>> output = pos_encoder(embedding)
        """

        return self.dropout(embedding + self.positional_encoding)
