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
        num_token: Number of tokens (required).
        segments: Number of segments for gradient checkpointing (required).

    Examples:
        >>> transformer = Transformer(d_model, data_len, dropout, ff, nhead,
                                    num_layer, num_token, segments)"""
    def __init__(
        self,
        d_model: int,
        data_len: int,
        dropout: float,
        ff: int,
        nhead: int,
        num_layer: int,
        num_token: int,
        segments: int,
    ) -> None:
        super().__init__()
        self.data_len = data_len
        self.embed = Embedding(d_model=d_model, num_token=num_token)
        self.bottleneck = nn.Linear(
            in_features=d_model,
            out_features=d_model,
        )
        self.encoder = nn.Sequential(*[
            RotaryTransformerLayer(
                d_model=d_model,
                dropout=dropout,
                ff=ff,
                nhead=nhead,
            ) for _ in range(num_layer)
        ])
        self.norm = nn.LayerNorm((d_model, ))
        mask = nn.Transformer.generate_square_subsequent_mask(data_len)
        self.register_buffer("mask", mask)
        self.linear = nn.Linear(
            in_features=d_model,
            out_features=num_token,
        )
        self.segments = segments

    def forward(self, data: Tensor) -> Tensor:
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
            - output: `(batch, num_token, data_len)`

        Examples:
            >>> data = torch.randint(
                ...     low=0,
                ...     high=num_token,
                ...     size=(batch_size, seq_len),
                ...     dtype=torch.int64,
                ... )
            >>> output = transformer(data)"""

        # Embed input
        embed = self.embed(data)
        # Pass through bottleneck
        bottleneck = self.bottleneck(embed)
        # Generate temporal data
        temporal = torch.arange(
            start=0,
            end=self.data_len,
            step=1,
            dtype=torch.float32,
            device=bottleneck.device,
        ).unsqueeze(dim=0).repeat((embed.size(0), 1))
        # Generate mask
        seq_len = embed.shape[1]
        if seq_len > self.data_len:
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
            mask = mask.to(bottleneck.device)
        else:
            mask = self.mask[:seq_len, :seq_len]
        # Pass through encoder
        # Use gradient checkpointing
        encode, _, _ = checkpoint_sequential(
            self.encoder,
            self.segments,
            (bottleneck, temporal, mask),
        )
        # Normalize
        normalize = self.norm(encode)
        # Pass through linear layer
        linear = self.linear(normalize)
        # Permute dimensions for compatibility with nn.CrossEntropyLoss
        return linear.permute(0, 2, 1)
