"""Implementation of Loss functions."""
import numpy as np
import torch
from torch import nn, Tensor


class SimpleLoss(nn.Module):
    """Implementation of cross entropy loss layer.

    A wrapper around nn.CrossEntropyLoss.
    Ignores index 0 (padding).

    Examples:
        >>> from model.loss import SimpleLoss
        >>> loss = SimpleLoss()"""
    def __init__(self) -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, logit: Tensor, target: Tensor) -> Tensor:
        """Calculates cross entropy loss.

        Args:
            logit: The output of the model (required).
            target: The provided answer (required).

        Returns:
            The averaged cross entropy loss of all targets.

        Shapes:
            - logit: `(batch, num_token, seq_len)`
            - target: `(batch, seq_len)`
            - output: `(1,)`

        Examples:
            >>> logit = torch.rand(
                ...     size=(batch, num_token, seq_len),
                ...     dtype=torch.float,
                ... )
            >>> target = torch.randint(
                ...     low=0,
                ...     high=num_token,
                ...     size=(batch, seq_len),
                ...     dtype=torch.int64,
                ... )
            >>> loss = loss(logit, target)"""
        return self.cross_entropy(logit, target)


class MultiResolutionTimeLoss(nn.Module):
    """Implementation of MultiResolutionTimeLoss.

    A collection of multiple loss layers.

    Args:
        level: The number of loss layers (required).
        res_max: The maximum resolution (required).
        res_min: The minimum resolution (required).
        max_time: The maximum time (required).

    Example:
        >>> loss = MultiResolutionTimeLoss(
            ...     level=16,
            ...     res_max=1024,
            ...     res_min=16,
            ...     max_time=8,
            ...     )"""
    def __init__(
        self,
        level: int,
        res_max: int,
        res_min: int,
        max_time: int,
    ) -> None:
        super().__init__()
        resolutions = np.round(
            np.exp(np.linspace(
                np.log(res_min),
                np.log(res_max),
                level,
            ))).astype(np.int64)
        self.res_list = resolutions.tolist()
        self.register_buffer("resolutions", torch.LongTensor(self.res_list))
        self.loss = nn.CrossEntropyLoss(reduction="none")
        denominators = (resolutions + max_time - 1) // max_time
        self.register_buffer("denominators", torch.LongTensor(denominators))

    def forward(self, logit: Tensor, fraction: Tensor) -> Tensor:
        """Calculates loss from logits and fractions.

        In each loss layer, the closest two indices are calculated from the
        fraction. The loss is calculated as the weighted average of the loss
        from the two indices, where the weights are the distance between the
        fraction and the indices. The loss is then averaged over all indices.

        Args:
            logit: The output of the model (required).
            fraction: The provided answer (required).

        Returns:
            The averaged loss of all indices.

        Shapes:
            logit: [..., sum(resolutions)]
            fraction: [..., level]

        Examples:
            >>> loss = loss(logit, fraction)"""
        scale = torch.einsum(
            "ij,k->ijk",
            fraction,
            self.denominators,
        )
        lower = torch.floor(scale).long()
        upper = torch.ceil(scale).long()
        lower = torch.minimum(lower, self.resolutions - 1)
        upper = torch.minimum(upper, self.resolutions - 1)
        logits = logit.split(self.res_list, dim=1)
        lower_loss = torch.stack(
            [
                self.loss(lower_logit, lower[..., i])
                for i, lower_logit in enumerate(logits)
            ],
            dim=-1,
        )
        upper_loss = torch.stack(
            [
                self.loss(upper_logit, upper[..., i])
                for i, upper_logit in enumerate(logits)
            ],
            dim=-1,
        )
        lower_weight = torch.where(
            lower == upper,
            torch.ones_like(scale),
            upper - scale,
        )
        upper_weight = torch.where(
            lower == upper,
            torch.zeros_like(scale),
            scale - lower,
        )
        loss = lower_loss * lower_weight + upper_loss * upper_weight
        return loss.mean()


class MusicLoss(nn.Module):
    """Implementation of MusicLoss.

    A collection of multiple loss layers.

    Args:
        duration_level: The number of layers in duration encoding (required).
        duration_resolution_max: The maximum resolution of the duration encoding (required).
        duration_resolution_min: The minimum resolution of the duration encoding (required).
        duration_max: The maximum time of the duration encoding (required).
        delta_level: The number of layers in delta encoding (required).
        delta_resolution_max: The maximum resolution of the delta encoding (required).
        delta_resolution_min: The minimum resolution of the delta encoding (required).
        delta_max: The maximum time of the delta encoding (required).

    Example:
        >>> loss = MusicLoss(
            ...     duration_level=16,
            ...     duration_resolution_max=1024,
            ...     duration_resolution_min=16,
            ...     duration_max=8,
            ...     delta_level=16,
            ...     delta_resolution_max=1024,
            ...     delta_resolution_min=16,
            ...     delta_max=2,
            ...     )"""
    def __init__(
        self,
        duration_level: int,
        duration_resolution_max: int,
        duration_resolution_min: int,
        duration_max: int,
        delta_level: int,
        delta_resolution_max: int,
        delta_resolution_min: int,
        delta_max: int,
    ) -> None:
        super().__init__()
        self.program_loss = SimpleLoss()
        self.note_loss = SimpleLoss()
        self.velocity_loss = SimpleLoss()
        self.duration_loss = MultiResolutionTimeLoss(
            level=duration_level,
            res_max=duration_resolution_max,
            res_min=duration_resolution_min,
            max_time=duration_max,
        )
        self.delta_loss = MultiResolutionTimeLoss(
            level=delta_level,
            res_max=delta_resolution_max,
            res_min=delta_resolution_min,
            max_time=delta_max,
        )

    def forward(
        self,
        program_logit: Tensor,
        note_logit: Tensor,
        velocity_logit: Tensor,
        duration_logit: Tensor,
        delta_logit: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Calculates loss from logits and targets.

        Args:
            program_logit: The output of the program layer (required).
            note_logit: The output of the note layer (required).
            velocity_logit: The output of the velocity layer (required).
            duration_logit: The output of the duration layer (required).
            delta_logit: The output of the delta layer (required).
            target: The provided answer (required).

        Returns:
            The summed loss of all components.

        Shapes:
            program_logit: [..., num_programs]
            note_logit: [..., num_notes]
            velocity_logit: [..., num_velocities]
            duration_logit: [..., sum(duration_resolutions)]
            delta_logit: [..., sum(delta_resolutions)]
            target: [..., 7]

        Examples:
            >>> loss = loss(
            ...     program_logit,
            ...     note_logit,
            ...     velocity_logit,
            ...     duration_logit,
            ...     delta_logit,
            ...     target,
            ... )"""
        program_loss = self.program_loss(program_logit, target[:, :, 0])
        note_loss = self.note_loss(note_logit, target[:, :, 1])
        velocity_loss = self.velocity_loss(velocity_logit, target[:, :, 2])
        duration = target[:, :, 3] / target[:, :, 4]
        duration_loss = self.duration_loss(duration_logit, duration)
        delta = target[:, :, 5] / target[:, :, 6]
        delta_loss = self.delta_loss(delta_logit, delta)
        return program_loss + note_loss + velocity_loss + duration_loss + delta_loss
