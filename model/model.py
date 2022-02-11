"""Implemnetation of MusicModel."""
from pytorch_lightning import LightningModule
import torch
from torch import Tensor

from model.accuracy import SimpleAccuracy
from model.loss import SimpleLoss
from model.transformer import Transformer


class MusicModel(LightningModule):
    """Implementation of MusicModel.

    Implements a Transformer model with rotary embedding.

    Args:
        d_model: The dimension of the model (required).
        data_len: The length of the input data (required).
        dropout: The dropout rate (required).
        ff: The number of feed-forward layers (required).
        lr: The learning rate (required).
        nhead: The number of heads in the multi-head attention (required).
        num_layer: The number of layers in the Transformer (required).
        num_token: The number of tokens (required).
        segments: The number of segments in gradient checkpointing (required).

    Example:
        >>> model = MusicModel(
            ...     d_model=512,
            ...     data_len=1024,
            ...     dropout=0.1,
            ...     ff=2048,
            ...     lr=0.001,
            ...     nhead=8,
            ...     num_layer=16,
            ...     num_token=1024,
            ...     segments=4,
            ... )"""
    def __init__(
        self,
        d_model: int,
        data_len: int,
        dropout: float,
        ff: int,
        lr: float,
        nhead: int,
        num_layer: int,
        num_token: int,
        segments: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = lr
        self.transformer = Transformer(
            d_model=d_model,
            data_len=data_len,
            dropout=dropout,
            ff=ff,
            nhead=nhead,
            num_layer=num_layer,
            num_token=num_token,
            segments=segments,
        )
        self.loss = SimpleLoss()
        self.acc = SimpleAccuracy()
        self.example_input_array = torch.ones(1, data_len, dtype=torch.int64)

    def forward(self, data: Tensor) -> Tensor:
        return self.transformer(data)

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(params={"lr": self.hparams.lr})

    def training_step(self, batch: Tensor, *args, **kwargs) -> Tensor:
        output = self(batch[:, :-1])
        loss = self.loss(output, batch[:, 1:])
        self.acc(output, batch[:, 1:])
        self.log("train/loss", loss)
        self.log("train/acc", self.acc.accuracy)
        return loss

    def validation_step(self, batch: Tensor, *args, **kwargs) -> Tensor:
        output = self(batch[:, :-1])
        loss = self.loss(output, batch[:, 1:])
        self.acc(output, batch[:, 1:])
        self.log("val/loss", loss)
        self.log("val/acc", self.acc.accuracy)
        return loss

    def test_step(self, batch: Tensor, *args, **kwargs) -> Tensor:
        output = self(batch[:, :-1])
        loss = self.loss(output, batch[:, 1:])
        self.acc(output, batch[:, 1:])
        self.log("test/loss", loss)
        self.log("test/acc", self.acc.accuracy)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,
            amsgrad=True,
        )
