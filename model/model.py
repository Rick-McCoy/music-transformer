"""
    Defines MusicModel class.
"""
import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torchmetrics import Accuracy

from config.config import CustomConfig
from model.loss import SimpleLoss
from model.transformer import Transformer


class MusicModel(LightningModule):
    """
        The MusicModel class.
        This implements various Lightningmodule methods.
    """
    def __init__(
        self, cfg: CustomConfig
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.learning_rate = cfg.learning_rate
        self.transformer = Transformer(
            d_model=cfg.d_model,
            data_len=cfg.data_len,
            dropout=cfg.dropout,
            ff=cfg.feed_forward,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            num_token=cfg.num_token,
            segments=cfg.segments,
        )
        self.loss = SimpleLoss()
        self.acc = Accuracy(top_k=1, ignore_index=0)
        self.example_input_array = torch.zeros(1, cfg.data_len, dtype=torch.int64)

    def forward(self, data: Tensor, *args, **kwargs) -> Tensor:
        return self.transformer(data)

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(params={"lr": self.hparams.lr})

    def training_step(self, batch: Tensor, *args, **kwargs) -> Tensor:
        output = self(batch[:, :-1])
        loss = self.loss(output, batch[:, 1:])
        self.acc(output, batch[:, 1:])
        self.log("train/loss", loss)
        self.log("train/acc", self.acc)
        return loss

    def validation_step(self, batch: Tensor, *args, **kwargs) -> Tensor:
        output = self(batch[:, :-1])
        loss = self.loss(output, batch[:, 1:])
        self.acc(output, batch[:, 1:])
        self.log("val/loss", loss)
        self.log("val/acc", self.acc)
        return loss

    def test_step(self, batch: Tensor, *args, **kwargs) -> Tensor:
        output = self(batch[:, :-1])
        loss = self.loss(output, batch[:, 1:])
        self.acc(output, batch[:, 1:])
        self.log("test/loss", loss)
        self.log("test/acc", self.acc)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.learning_rate, amsgrad=True
        )
