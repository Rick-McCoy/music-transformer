import math

import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    """
    Inject some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as
    the embeddings, so that the two can be summed. Here, we use sine and cosine
    functions of different frequencies.

    Args:
        d_model: The embedding size. (required)
        data_len: The sequence length. (required)
        dropout: The dropout rate. (required)
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model, data_len, dropout)
    """

    def __init__(self, d_model: int, data_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, data_len, dtype=torch.float).reshape(1, -1, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        term = position * div_term
        positional_encoding = torch.flatten(
            torch.stack([torch.sin(term), torch.cos(term)], dim=-1), start_dim=-2
        )
        self.register_buffer("positional_encoding", positional_encoding)
        self.positional_encoding: Tensor

    def forward(self, embedding: Tensor):
        """Inputs of forward function
        Args:
            embedding: The tensor to be positional encoded. (required)
        Shape:
            embedding: [batch_size, data_len, d_model]
            output: [batch_size, data_len, d_model]
        Outputs:
            The positional encoded tensor.
        Examples:
            >>> output = pos_encoder(embedding)
        """

        return self.dropout(embedding + self.positional_encoding)
