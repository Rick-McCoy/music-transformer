import math

from omegaconf import DictConfig
import torch
from torch import nn, Tensor


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.dropout = nn.Dropout(p=cfg.model.dropout)
        self.embed_dim = cfg.model.d_model * 3
        arange = torch.arange(0, self.embed_dim, 2, dtype=torch.float32)
        self.register_buffer("arange", arange)
        self.arange: Tensor

    def forward(self, embedding: Tensor, time: Tensor):
        r"""Inputs of forward function
        Args:
            embedding: the sequence fed to the positional encoder model (required).
        Shape:
            embedding: [batch size, sequence length, model dim * 3]
            time: [batch size, sequence length]
            output: [batch size, sequence length, model dim * 3]
        Examples:
            >>> output = pos_encoder(embedding)
        """

        with torch.no_grad():
            position = time.unsqueeze(-1)
            div_term = torch.exp(self.arange *
                                 (-math.log(10000.0) / self.embed_dim))
            term = position * div_term
            positional_encoding = torch.stack(
                [torch.sin(term), torch.cos(term)],
                dim=-1).reshape_as(embedding)

        return self.dropout(embedding + positional_encoding)
