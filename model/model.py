"""
    Defines MusicModel class.
"""

import torch
import torchmetrics
from pytorch_lightning import LightningModule
from torch import Tensor

from model.loss import CrossEntropy
from model.transformer import Transformer


class MusicModel(LightningModule):
    """
    The MusicModel class.
    This implements various Lightningmodule methods.
    """

    def __init__(
        self,
        learning_rate: float,
        d_model: int,
        data_len: int,
        dropout: float,
        feed_forward: int,
        nhead: int,
        num_layers: int,
        num_tokens: int,
        is_training: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.transformer = Transformer(
            d_model=d_model,
            data_len=data_len,
            dropout=dropout,
            feed_forward=feed_forward,
            nhead=nhead,
            num_layers=num_layers,
            num_tokens=num_tokens,
        )
        self.loss = CrossEntropy()
        self.acc = torchmetrics.Accuracy(top_k=1, ignore_index=0, mdmc_average="global")
        self.example_input_array = torch.zeros(1, data_len, dtype=torch.int64)
        self.is_training = is_training

    def forward(self, data: Tensor) -> Tensor:  # pylint: disable=arguments-differ
        return self.transformer(data)

    def training_step(self, *args, **kwargs) -> Tensor:  # pylint: disable=unused-argument
        batch = args[0]
        output = self(batch[:, :-1])
        loss = self.loss(output, batch[:, 1:])
        if self.is_training:
            self.acc(output, batch[:, 1:])
            self.log("train/loss", loss)
            self.log("train/acc", self.acc)
        return loss

    def validation_step(self, *args, **kwargs) -> Tensor:  # pylint: disable=unused-argument
        batch = args[0]
        output = self(batch[:, :-1])
        loss = self.loss(output, batch[:, 1:])
        self.acc(output, batch[:, 1:])
        self.log("val/loss", loss)
        self.log("val/acc", self.acc)
        return loss

    def test_step(self, *args, **kwargs) -> Tensor:  # pylint: disable=unused-argument
        batch = args[0]
        output = self(batch[:, :-1])
        loss = self.loss(output, batch[:, 1:])
        self.acc(output, batch[:, 1:])
        self.log("test/loss", loss)
        self.log("test/acc", self.acc)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.learning_rate, amsgrad=True)
