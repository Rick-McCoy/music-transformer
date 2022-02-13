"""Implementation of Embedding."""
import numpy as np
import torch
from torch import nn, Tensor


class Embedding(nn.Module):
    """Implementation of Embedding.

    A wrapper around nn.Embedding.
    Ignores index 0 (padding).

    Args:
        d_model: The dimension of the model (required).
        num_token: The number of tokens (required).

    Example:
        >>> from model.embedding import Embedding
        >>> embedding = Embedding(d_model=512, num_token=128)"""
    def __init__(self, d_model: int, num_token: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=num_token,
                                  embedding_dim=d_model,
                                  padding_idx=0)

    def forward(self, data: Tensor):
        """Calculates embedding from indices.

        Args:
            data: Int64 indices (required).

        Returns:
            The embedding of the indices.

        Shapes:
            - data: `(batch, seq_len)`
            - output: `(batch, seq_len, d_model)`

        Examples:
            >>> data = torch.randint(
                ...     low=0,
                ...     high=128,
                ...     size=(2, 10),
                ...     dtype=torch.int64)
            >>> embedding = Embedding(d_model=512, num_token=128)
            >>> embed = embedding(data)"""
        return self.embed(data)


class MultiResolutionTimeEmbedding(nn.Module):
    """Implementation of MultiResolutionTimeEmbedding.

    A collection of multiple embedding layers.

    Args:
        d_model: The dimension of the model (required).
        level: The number of embedding layers (required).
        res_max: The maximum resolution (required).
        res_min: The minimum resolution (required).
        max_time: The maximum time (required).

    Example:
        >>> embedding = MultiResolutionTimeEmbedding(
            ...     d_model=512, level=16, res_max=128, res_min=8, max_time=4)"""
    def __init__(self, d_model: int, level: int, res_max: int, res_min: int,
                 max_time: int) -> None:
        super().__init__()
        resolutions = np.round(
            np.exp(np.linspace(np.log(res_min), np.log(res_max),
                               level))).astype(np.int64)
        self.register_buffer("resolutions", torch.LongTensor(resolutions))
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=resolution,
                         embedding_dim=d_model // level)
            for resolution in resolutions
        ])
        denominators = torch.LongTensor([
            (resolution + max_time - 1) // max_time
            for resolution in resolutions
        ])
        self.register_buffer("denominators", denominators)
        self.level = level

    def forward(self, fraction: Tensor) -> Tensor:
        """Calculates embedding from fractions.

        In each embedding layer, the closest two indices are calculated from the
        fraction. The embedding is then calculated as the weighted average of
        the embeddings of the two indices, where the weight is the distance
        between the fraction and the indices. The embeddings are then concatenated.

        Args:
            fraction: Float32 fractions (required).

        Returns:
            The concatenated embeddings.

        Shapes:
            fraction: [num_batches, sequence_length]
            output: [num_batches, sequence_length, d_model]

        Examples:
            >>> embedding = embed(fraction)"""
        scale = torch.einsum("ij,k->ijk", fraction, self.denominators)
        lower = torch.floor(scale).long()
        upper = torch.ceil(scale).long()
        lower = torch.minimum(lower, self.resolutions - 1)
        upper = torch.minimum(upper, self.resolutions - 1)
        lower_embed = torch.stack(
            [embed(lower[..., i]) for i, embed in enumerate(self.embeddings)],
            dim=-1)
        upper_embed = torch.stack(
            [embed(upper[..., i]) for i, embed in enumerate(self.embeddings)],
            dim=-1)
        lower_weight = torch.where(lower == upper, torch.ones_like(scale),
                                   upper - scale).unsqueeze(dim=-2)
        upper_weight = torch.where(lower == upper, torch.zeros_like(scale),
                                   scale - lower).unsqueeze(dim=-2)
        embedding = lower_embed * lower_weight + upper_embed * upper_weight
        return embedding.flatten(start_dim=-2)


class MusicEmbedding(nn.Module):
    """Implementation of MusicEmbedding.

    A collection of multiple embedding modules.
    The program embedding is a one-hot embedding.
    The note embedding is a one-hot embedding.
    The velocity embedding is a one-hot embedding.
    The duration embedding is a multi-resolution temporal embedding.
    The delta embedding is a multi-resolution temporal embedding.

    Args:
        d_model: The dimension of the model (required).
        duration_level: The number of levels in the duration encoding (required).
        duration_resolution_max: The maximum resolution of the duration encoding (required).
        duration_resolution_min: The minimum resolution of the duration encoding (required).
        duration_max: The maximum time of the duration encoding (required).
        delta_level: The number of levels in the delta encoding (required).
        delta_resolution_max: The maximum resolution of the delta encoding (required).
        delta_resolution_min: The minimum resolution of the delta encoding (required).
        delta_max: The maximum time of the delta encoding (required).
        num_program: The number of program tokens (required).
        num_note: The number of note tokens (required).
        num_velocity: The number of velocity tokens (required).
        num_special: The number of special tokens (required).

    Example:
        >>> embedding = MusicEmbedding(
            ...     d_model=512,
            ...     duration_level=16,
            ...     duration_resolution_max=1024,
            ...     duration_resolution_min=16,
            ...     duration_max=8,
            ...     delta_level=16,
            ...     delta_resolution_max=1024,
            ...     delta_resolution_min=16,
            ...     delta_max=2,
            ...     num_program=129,
            ...     num_note=128,
            ...     num_velocity=128,
            ...     num_special=3,
            ... )"""
    def __init__(self, d_model: int, duration_level: int,
                 duration_resolution_max: int, duration_resolution_min: int,
                 duration_max: int, delta_level: int,
                 delta_resolution_max: int, delta_resolution_min: int,
                 delta_max: int, num_program: int, num_note: int,
                 num_velocity: int, num_special: int) -> None:
        super().__init__()
        self.program_embedding = Embedding(d_model=d_model,
                                           num_token=num_program + num_special)
        self.note_embedding = Embedding(d_model=d_model,
                                        num_token=num_note + num_special)
        self.velocity_embedding = Embedding(d_model=d_model,
                                            num_token=num_velocity +
                                            num_special)
        self.duration_embedding = MultiResolutionTimeEmbedding(
            d_model=d_model,
            level=duration_level,
            res_max=duration_resolution_max,
            res_min=duration_resolution_min,
            max_time=duration_max)
        self.delta_embedding = MultiResolutionTimeEmbedding(
            d_model=d_model,
            level=delta_level,
            res_max=delta_resolution_max,
            res_min=delta_resolution_min,
            max_time=delta_max)

    def forward(self, data: Tensor) -> Tensor:
        """Calculates embedding from data.

        The first three tokens are program, note, and velocity.
        The next two tokens are the numerator and denominator of duration.
        The next two tokens are the numerator and denominator of delta.

        Args:
            data: Int64 indices (required).

        Returns:
            The concatenated embeddings.

        Shapes:
            data: [num_batches, sequence_length, 7]
            output: [num_batches, sequence_length, d_model * 5]

        Examples:
            >>> embedding = embed(data)"""
        program = data[:, :, 0]
        note = data[:, :, 1]
        velocity = data[:, :, 2]
        duration = data[:, :, 3] / data[:, :, 4]
        delta = data[:, :, 5] / data[:, :, 6]
        program_embed = self.program_embedding(program)
        note_embed = self.note_embedding(note)
        velocity_embed = self.velocity_embedding(velocity)
        duration_embed = self.duration_embedding(duration)
        delta_embed = self.delta_embedding(delta)
        embed = torch.cat([
            program_embed, note_embed, velocity_embed, duration_embed,
            delta_embed
        ],
                          dim=-1)
        return embed
