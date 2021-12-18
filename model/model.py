from omegaconf import DictConfig
from pytorch_lightning import LightningModule
import torch
from torch import Tensor
from torchmetrics import Accuracy

from model.loss import SimpleLoss
from model.transformer import Transformer


class MusicModel(LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.learning_rate = cfg.train.lr
        self.transformer = Transformer(cfg)
        self.loss = SimpleLoss(cfg)
        self.acc = Accuracy(top_k=1)
        self.example_input_array = torch.zeros(cfg.train.batch_size,
                                               cfg.model.data_len,
                                               dtype=torch.int64)

    def forward(self, data: Tensor, *args, **kwargs) -> Tensor:
        return self.transformer(data)

    def training_step(self, batch: Tensor, *args, **kwargs) -> Tensor:
        output = self(batch[:, :-1])
        loss = self.loss(output, batch[:, 1:])
        self.acc(output, batch[:, 1:])
        self.log("train_loss", loss)
        self.log("train_acc", self.acc)
        return loss

    def validation_step(self, batch: Tensor, *args, **kwargs) -> Tensor:
        output = self(batch[:, :-1])
        loss = self.loss(output, batch[:, 1:])
        self.acc(output, batch[:, 1:])
        self.log("val_loss", loss)
        self.log("val_acc", self.acc)
        return loss

    def test_step(self, batch: Tensor, *args, **kwargs) -> Tensor:
        output = self(batch[:, :-1])
        loss = self.loss(output, batch[:, 1:])
        self.acc(output, batch[:, 1:])
        self.log("test_loss", loss)
        self.log("test_acc", self.acc)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(),
                                lr=self.learning_rate,
                                amsgrad=True)
