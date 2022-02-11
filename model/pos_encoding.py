"""Positional Encoding implementation.

Original implementation link:
https://github.com/pytorch/examples/blob/master/word_language_model/model.py
Modified significantly by me."""
import math

import torch
from torch import nn, Tensor


class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.

    Args:
        d_model: The embed dim (required).
        dropout: The dropout value (required).
        max_len: The maximum length of the incoming sequence (required).

    Examples:
        >>> pos_encoding = PositionalEncoding(d_model, data_len, dropout)
    """
    def __init__(self, d_model: int, data_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.data_len = data_len
        positional_encoding = self.get_encoding(data_len)
        self.register_buffer("positional_encoding", positional_encoding)
        self.positional_encoding: Tensor

    def get_encoding(self, length: int) -> Tensor:
        """Calculates sinusoidal positional encoding.

        Args:
            length: Length of input sequence (required)

        Returns:
            The calculated positional encoding. The results are as follows:

            >>> pos_enc[pos, 2 * i] = sin(pos / pow(10000, 2 * i / d_model))
            >>> pos_enc[pos, 2 * i + 1] = cos(pos / pow(10000, 2 * i / d_model))

        Shapes:
            pos_enc: [data_len, d_model]

        Examples:
            >>> pos_enc = calculate_encoding(length)"""
        position = torch.arange(0, length, dtype=torch.float32)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / self.d_model))
        term = torch.einsum("i,j->ij", position, div_term).unsqueeze(dim=-1)
        positional_encoding = torch.stack(
            [torch.sin(term), torch.cos(term)],
            dim=-1,
        ).flatten(start_dim=1)
        return positional_encoding

    def forward(self, sequence: Tensor) -> Tensor:
        """Performs encoding & dropout on input.

        Args:
            sequence: The sequence fed to the positional encoder (required).

        Returns:
            The sequence, with positional encoding added and dropout performed.

        Shapes:
            sequence: [batch_size, data_len, d_model]
            output: [batch_size, data_len, d_model]

        Examples:
            >>> output = pos_encoding(sequence)
        """

        if sequence.shape[1] != self.data_len:
            pos_encoding = self.get_encoding(sequence.shape[1]).to(sequence)
        else:
            pos_encoding = self.positional_encoding

        return self.dropout(sequence + pos_encoding)
