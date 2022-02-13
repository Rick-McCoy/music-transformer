"""Implements various attention mechanisms."""
import math
from typing import Tuple

import torch
from torch import nn, Tensor


class RotaryAttention(nn.Module):
    """Implements Rotary Attention.

    Rotary Attention is a variant of Multi-Head Attention that uses
    a sinusoidal frequency for positional encoding.
    In order to encode a vector q, we first view it as a complex vector Q
    where if `q = [q1, q2, ...], Q = [q1 + i * q2, q3 + i * q4, ...]`.
    Then, the encoded rotary vector is:
    ```
    f(Q, m) = Q * exp(i * m * freq)
    = [(q1 + i * q2) * exp(i * m * freq1), (q3 + i * q4) * exp(i * m * freq2), ...]
    = [(q1 * cos(m * freq1) - q2 * sin(m * freq1))
    + i * (q1 * sin(m * freq1) + q2 * cos(m * freq1)), ...]
    ```
    where `freq` is a vector of frequencies.
    However, since computing complex numbers is expensive, we use two columns
    of real numbers to represent the complex vector.
    Therefore the encoded rotary vector is:
    ```
    f(q, m) = [q1 * cos(m * freq1) - q2 * sin(m * freq1),
               q1 * sin(m * freq1) + q2 * cos(m * freq1), ...]
    ```
    This can be computed by the following equation:
    ```
    rot(q) = [-q2, q1, -q4, q3, ...]
    f(q, m) = q * cos(repeat(m * freq, 2)) + rot(q) * sin(repeat(m * freq, 2))
    ```

    Similar to the original positional encoding, we use an
    exponential-sinusoidal function for frequency:
    ```
    freq = 10000^(range(0, dim, 2) / dim)
    ```
    Where `dim` is `d_model // 2`.

    Args:
        d_model: Dimension of the model (required).
        nhead: Number of heads (required).
        num_temp: Number of temporal columns (required).
        dropout: Dropout probability (required).

    Examples:
        >>> from model.attention import RotaryAttention
        >>> rotary_attention = RotaryAttention(d_model=512, nhead=8, num_temp=4, dropout=0.1)"""
    def __init__(self, d_model: int, nhead: int, num_temp: int,
                 dropout: float) -> None:
        super().__init__()
        self.query_linear = nn.Linear(in_features=d_model,
                                      out_features=d_model)
        self.key_linear = nn.Linear(in_features=d_model, out_features=d_model)
        self.value_linear = nn.Linear(in_features=d_model,
                                      out_features=d_model)
        self.dropout = nn.Dropout(p=dropout)
        dim = d_model // nhead // num_temp
        freq = torch.pow(10000, -torch.arange(0, dim, 2) / dim)
        freq = torch.repeat_interleave(freq, repeats=2, dim=0)
        self.register_buffer("freq", freq)
        self.nhead = nhead
        self.scale = math.sqrt(d_model // nhead)

    def apply_rotary(self, data: Tensor, sine: Tensor,
                     cosine: Tensor) -> Tensor:
        """Applies rotary embedding to data.

        Args:
            data: Input data (required).
            temporal: Tenmporal data (required).

        Returns:
            Rotary embeddings.

        Shapes:
            - data: `(batch_size, seq_len, nhead, d_model // nhead)`
            - sine: `(batch_size, seq_len, d_model // nhead)`
            - cosine: `(batch_size, seq_len, d_model // nhead)`
            - output: `(batch_size, seq_len, nhead, d_model // nhead)`

        Examples:
            >>> data = torch.randn(batch_size, seq_len, nhead, d_model // nhead)
            >>> sine = torch.randn(batch_size, seq_len, d_model // nhead)
            >>> cosine = torch.randn(batch_size, seq_len, d_model // nhead)
            >>> output = rotary_attention.apply_rotary(data, sine, cosine)"""

        # Rotate query
        # Shape: `(batch_size, seq_len, nhead, d_model // nhead)`
        stack_data = torch.stack([-data[..., 1::2], data[..., 0::2]], dim=-1)
        rotate_data = torch.flatten(stack_data, start_dim=-2)

        # Calculate rotary embedding
        # Shape: `(batch_size, seq_len, nhead, d_model // nhead)`
        rotary_data = torch.einsum("blhs,bls->blhs", data, cosine) + \
            torch.einsum("blhs,bls->blhs", rotate_data, sine)

        return rotary_data

    def forward(self, data: Tensor, temporal: Tensor, mask: Tensor) -> Tensor:
        """Forward pass.

        Args:
            data: Input data (required).
            temporal: Temporal data (required).
            mask: Mask for the input (required).

        Returns:
            Attention output.

        Shapes:
            - data: `(batch_size, seq_len, d_model)`
            - temporal: `(batch_size, seq_len, num_temp)`
            - mask: `(seq_len, seq_len)`
            - output: `(batch_size, seq_len, d_model)`

        Examples:
            >>> data = torch.randn(batch_size, seq_len, d_model)
            >>> temporal = torch.randn(batch_size, seq_len, num_temp)
            >>> mask = torch.zeros(seq_len, seq_len)
            >>> output, _, _ = rotary_attention(data, temporal, mask)"""

        # Calculate query, key, value
        # Shape: `(batch_size, seq_len, d_model)`
        query = self.query_linear(data)
        key = self.key_linear(data)
        value = self.value_linear(data)

        # Unflatten heads
        # Shape: `(batch_size, seq_len, nhead, d_model // nhead)`
        query = query.unflatten(-1, (self.nhead, -1))
        key = key.unflatten(-1, (self.nhead, -1))
        value = value.unflatten(-1, (self.nhead, -1))

        # Calculate frequency
        # Shape: `(batch_size, seq_len, num_temp, d_model // nhead // num_temp)`
        freqs = torch.einsum("blt,d->bltd", temporal, self.freq)

        # Flatten frequency
        # Shape: `(batch_size, seq_len, d_model // nhead)`
        freqs = torch.flatten(freqs, start_dim=-2)

        # Calculate cosine and sine
        # Shape: `(batch_size, seq_len, d_model // nhead)`
        cosine = torch.cos(freqs)
        sine = torch.sin(freqs)

        # Rotate query and key
        rotary_query = self.apply_rotary(query, sine, cosine)
        rotary_key = self.apply_rotary(key, sine, cosine)

        # Calculate attention
        # Shape: `(batch_size, nhead, seq_len, seq_len)`
        logits = torch.einsum("bqhs,bkhs->bhqk", rotary_query, rotary_key)
        # Scale logits
        logits = logits / self.scale
        # Mask out invalid positions
        # Since mask is of shape `(seq_len, seq_len)`, broadcasting is automatic
        logits += mask

        # Calculate attention weights
        # Shape: `(batch_size, nhead, seq_len, seq_len)`
        weights = torch.softmax(logits, dim=-1)
        # Dropout
        weights = self.dropout(weights)

        # Calculate context
        # Shape: `(batch_size, seq_len, nhead, d_model // nhead)`
        context = torch.einsum("bhqk,bkhs->bqhs", weights, value)
        # Flatten heads
        # Shape: `(batch_size, seq_len, d_model)`
        context = context.flatten(start_dim=-2)

        return context


class RotaryTransformerLayer(nn.Module):
    """Implementation of RotaryTransformerLayer.

    The attention operation is performed with norm_first.
    Therefore the operation is as follows:
    - output1 = input + dropout(attention(norm(input)))
    - output2 = output1 + dropout(FF(RELU(FF(norm(output1)))))

    Temporal & Mask are forwarded for gradient checkponting.

    Args:
        d_model: Transformer hidden dimension size (required).
        dropout: Dropout probability (required).
        ff: Feed-forward dimension size (required).
        nhead: Number of heads (required).
        num_temp: Number of temporal columns (required).

    Examples:
        >>> from model.attention import RotaryTransformerLayer
        >>> rotary_transformer_layer = RotaryTransformerLayer(
            ...     d_model=512, dropout=0.1, ff=2048,
            ...     nhead=8, num_temp=4)"""
    def __init__(self, d_model: int, dropout: float, ff: int, nhead: int,
                 num_temp: int) -> None:
        super().__init__()
        self.attention = RotaryAttention(d_model=d_model,
                                         nhead=nhead,
                                         dropout=dropout,
                                         num_temp=num_temp)
        self.linear1 = nn.Linear(in_features=d_model, out_features=ff)
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(in_features=ff, out_features=d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(
            self, batch: Tuple[Tensor, Tensor,
                               Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass.

        Args:
            batch: Tuple of (data, temporal, mask) (required).

        Returns:
            Tuple of (output, temporal, mask).

        Shapes:
            - data: `(batch_size, seq_len, d_model)`
            - temporal: `(batch_size, seq_len, num_temp)`
            - mask: `(seq_len, seq_len)`
            - output: `(batch_size, seq_len, d_model)`

        Examples:
            >>> data = torch.randn(batch_size, seq_len, d_model)
            >>> temporal = torch.randn(batch_size, seq_len, num_temp)
            >>> mask = torch.zeros(seq_len, seq_len)
            >>> output, temporal, mask = rotary_transformer_layer(
                ...     (data, temporal, mask))"""
        data, temporal, mask = batch
        attention = data + self.dropout1(
            self.attention(self.norm1(data), temporal, mask))
        feedforward = data + self.dropout2(
            self.linear2(
                self.dropout(torch.relu(self.linear1(self.norm2(attention))))))
        return feedforward, temporal, mask


class TransformerLayer(nn.Module):
    """Implementation of Transformer encoder layer.

    A wrapper around `nn.TransformerEncoderLayer`.
    Mask forwarding implemented for `nn.Sequential`.

    Args:
        d_model: Transformer hidden dimension size (required).
        dropout: Dropout probability (required).
        ff: Transformer feedforward dimension size (required).
        nhead: Number of Transformer heads (required).

    Examples:
        >>> from model.attention import TransformerLayer
        >>> transformer_layer = TransformerLayer(
            ...     d_model=512, dropout=0.1, ff=2048, nhead=8)"""
    def __init__(self, d_model: int, dropout: float, ff: int,
                 nhead: int) -> None:
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                nhead=nhead,
                                                dim_feedforward=ff,
                                                dropout=dropout,
                                                batch_first=True,
                                                norm_first=True)

    def forward(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """Calculates a single transformer encoder layer.

        Since saving a float32 mask for every layer unnecessarily wastes memory,
        we forward the same mask across all layers.
        This enables `nn.Sequential` grouping and consequently gradient checkpointing.

        Args:
            batch: A tuple of input data & causal mask (required).

        Results:
            The encoded data and the same mask.

        Shapes:
            - data: `(batch_size, seq_len, d_model)`
            - mask: `(seq_len, seq_len)`
            - output: `(batch_size, seq_len, d_model)`

        Examples:
            >>> data = torch.randn(batch_size, seq_len, d_model)
            >>> mask = torch.zeros(seq_len, seq_len)
            >>> output, mask = transformer_layer((data, mask))"""

        # Unpack batch
        data, mask = batch
        # Forward layer
        output = self.layer(data, src_mask=mask)
        # Return output & mask
        return output, mask
