"""Implementation of SimpleAccuracy."""
from torch import nn, Tensor
from torchmetrics import Accuracy


class SimpleAccuracy(nn.Module):
    """Implementation of the most basic accuracy function.

    Uses torchmetrics.Accuracy.
    Ignores index 0 (padding).

    Examples:
        >>> from model.accuracy import SimpleAccuracy
        >>> acc = SimpleAccuracy()"""
    def __init__(self) -> None:
        super().__init__()
        self.accuracy = Accuracy(top_k=1, ignore_index=0)

    def forward(self, logit: Tensor, target: Tensor) -> Tensor:
        """Calculates the accuracy of prediction agains provided target.

        Args:
            logit: The output of the model (required).
            target: The provided answer (required).

        Returns:
            The averaged accuracy of all targets.

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
            >>> accuracy = acc(logit, target)"""
        return self.accuracy(logit, target)


class MusicAccuracy(nn.Module):
    """Implementation of music accuracy.

    Calculates and averages the accuracy of program, note, and velocity.
    Note that the internal torchmetrics modules do not produce an output."""
    def __init__(self) -> None:
        super().__init__()
        self.program_accuracy = Accuracy(top_k=1, ignore_index=0)
        self.note_accuracy = Accuracy(top_k=1, ignore_index=0)
        self.velocity_accuracy = Accuracy(top_k=1, ignore_index=0)
        self.aggregate_accuracy = (self.program_accuracy + self.note_accuracy +
                                   self.velocity_accuracy) / 3

    def forward(
        self,
        program_logit: Tensor,
        note_logit: Tensor,
        velocity_logit: Tensor,
        target: Tensor,
    ) -> None:
        """Calculates and averages program, note, and velocity accuracy.

        The first column of target is program.
        The second column of target is note.
        The third column of target is velocity.

        Args:
            program_logit: The output of the program layer (required).
            note_logit: The output of the note layer (required).
            velocity_logit: The outpout of the velocity layer (required).
            target: The provided answer (required)."""
        self.program_accuracy(program_logit, target[:, :, 0])
        self.note_accuracy(note_logit, target[:, :, 1])
        self.velocity_accuracy(velocity_logit, target[:, :, 2])
