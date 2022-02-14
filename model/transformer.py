"""Implementation of Transformer."""
import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint_sequential

from model.attention import RotaryTransformerLayer
from model.embedding import Embedding


class Transformer(nn.Module):
    """Implementation of core model.

    Combines Embedding, PositionalEncoding, TransformerLayer into one.
    All trainable layers are contained here.

    Args:
        d_model: Transformer hidden dimension size (required).
        data_len: Input length, not necessarily enforced (required).
        dropout: Dropout probability (required).
        ff: Transformer feedforward dimension size (required).
        nhead: Number of Transformer heads (required).
        num_layer: Number of Transformer layers (required).
        num_pos: Number of position columns (required).
        num_token: Number of tokens (required).
        segments: Number of segments for gradient checkpointing (required).

    Examples:
        >>> transformer = Transformer(
            ...     d_model=128,
            ...     data_len=1024,
            ...     dropout=0.1,
            ...     ff=2048,
            ...     nhead=8,
            ...     num_layer=16,
            ...     num_pos=4,
            ...     num_token=1024,
            ...     segments=4,
            ... )"""
    def __init__(self, d_model: int, data_len: int, dropout: float, ff: int,
                 nhead: int, num_layer: int, num_pos: int, num_token: int,
                 segments: int) -> None:
        super().__init__()
        self.data_len = data_len
        self.embed = Embedding(d_model=d_model, num_token=num_token)
        self.bottleneck = nn.Linear(in_features=d_model, out_features=d_model)
        self.encoder = nn.Sequential(*[
            RotaryTransformerLayer(d_model=d_model,
                                   dropout=dropout,
                                   ff=ff,
                                   nhead=nhead,
                                   num_pos=num_pos) for _ in range(num_layer)
        ])
        self.norm = nn.LayerNorm((d_model, ))
        mask = nn.Transformer.generate_square_subsequent_mask(data_len)
        self.register_buffer("mask", mask)
        self.linear = nn.Linear(in_features=d_model, out_features=num_token)
        self.segments = segments

    def forward(self, data: Tensor, position: Tensor) -> Tensor:
        """Passes input sequence through Transformer.

        The cached causal mask is used when applicable.
        Else, a new mask is generated.
        Gradient checkpointing is used within the Transformer encoder.

        Args:
            data: Int64 input sequence (required).

        Results:
            Logits for the prediction of following tokens.

        Shapes:
            - data: `(batch, seq_len)`
            - position: (`(batch, seq_len, num_pos)`,
                         `(batch, seq_len, num_pos)`,
                         `(batch, seq_len, num_pos)`)
            - output: `(batch, num_token, data_len)`

        Examples:
            >>> data = torch.randint(
                ...     low=0,
                ...     high=num_token,
                ...     size=(batch_size, seq_len),
                ...     dtype=torch.int64,
                ... )
            >>> position = (torch.randint(
                ...     low=0,
                ...     high=num_pos,
                ...     size=(batch_size, seq_len, num_pos),
                ...     dtype=torch.int64,
                ... ) for _ in range(3))
            >>> output = transformer(data, position)"""

        # Get sequence length
        seq_len = data.shape[1]
        # Embed input
        embed = self.embed(data)
        # Pass through bottleneck
        bottleneck = self.bottleneck(embed)
        # Generate integer positions
        arange = torch.arange(start=0,
                              end=seq_len,
                              dtype=torch.float32,
                              device=bottleneck.device).unsqueeze(
                                  dim=0).unsqueeze(dim=-1).repeat(
                                      (embed.size(0), 1, 1))
        # Stack arange and position
        position = torch.cat((arange, position), dim=-1)
        # Get mask
        if seq_len > self.data_len:
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
            mask = mask.to(bottleneck.device)
        else:
            mask = self.mask[:seq_len, :seq_len]
        # Pass through encoder
        # Use gradient checkpointing
        encode, _, _ = checkpoint_sequential(self.encoder, self.segments,
                                             (bottleneck, position, mask))
        # Normalize
        normalize = self.norm(encode)
        # Pass through linear layer
        linear = self.linear(normalize)
        # Permute dimensions for compatibility with nn.CrossEntropyLoss
        return linear.permute(0, 2, 1)
