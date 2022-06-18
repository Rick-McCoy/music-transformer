"""
    Defines MusicModel class.
"""
import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torchmetrics import Accuracy

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
        segments: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.transformer = Transformer(
            d_model=d_model,
            data_len=data_len,
            dropout=dropout,
            ff=feed_forward,
            nhead=nhead,
            num_layers=num_layers,
            num_tokens=num_tokens,
            segments=segments,
        )
        self.loss = CrossEntropy()
        self.acc = Accuracy(top_k=1, ignore_index=0)
        self.example_input_array = torch.zeros(1, data_len, dtype=torch.int64)

    def forward(self, data: Tensor) -> Tensor:
        return self.transformer(data)

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(params={"lr": self.hparams.lr})

    def training_step(self, batch: Tensor) -> Tensor:
        output = self(batch[:, :-1])
        loss = self.loss(output, batch[:, 1:])
        self.acc(output, batch[:, 1:])
        self.log("train/loss", loss)
        self.log("train/acc", self.acc)
        return loss

    def validation_step(self, batch: Tensor) -> Tensor:
        output = self(batch[:, :-1])
        loss = self.loss(output, batch[:, 1:])
        self.acc(output, batch[:, 1:])
        self.log("val/loss", loss)
        self.log("val/acc", self.acc)
        return loss

    def test_step(self, batch: Tensor) -> Tensor:
        output = self(batch[:, :-1])
        loss = self.loss(output, batch[:, 1:])
        self.acc(output, batch[:, 1:])
        self.log("test/loss", loss)
        self.log("test/acc", self.acc)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.learning_rate, amsgrad=True)
