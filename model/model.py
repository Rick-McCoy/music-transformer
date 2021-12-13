from typing import List

import torch
from torch import Tensor
from torchmetrics import Accuracy
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

from model.classifier import SimpleClassifier
from model.loss import SimpleLoss


class SimpleModel(LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.learning_rate = cfg.train.lr
        self.classifier = SimpleClassifier(cfg)
        self.loss = SimpleLoss(cfg)
        self.acc = Accuracy(top_k=1)
        self.example_input_array = torch.zeros(
            (1, cfg.model.input_channels, cfg.model.h, cfg.model.w))

    def forward(self, data: Tensor, *args, **kwargs) -> Tensor:
        return self.classifier(data)

    def training_step(self, batch: List[Tensor], *args, **kwargs) -> Tensor:
        data, label = batch
        output = self(data)
        loss = self.loss(output, label)
        self.acc(output, label)
        self.log("train_loss", loss)
        self.log("train_acc", self.acc)
        return loss

    def validation_step(self, batch: List[Tensor], batch_idx: int, *args,
                        **kwargs) -> Tensor:
        data, label = batch
        output = self(data)
        loss = self.loss(output, label)
        self.acc(output, label)
        self.log("val_loss", loss)
        self.log("val_acc", self.acc)
        pred = torch.argmax(output, dim=-1)
        if batch_idx == 0:
            self.logger.experiment[0].add_images("val_img", data, 0)
            self.logger.experiment[0].add_text(
                "val_pred",
                ", ".join(str(i) for i in pred.cpu().detach().numpy()), 0)

    def test_step(self, batch: List[Tensor], batch_idx: int, *args,
                  **kwargs) -> Tensor:
        data, label = batch
        output = self(data)
        loss = self.loss(output, label)
        self.acc(output, label)
        self.log("test_loss", loss)
        self.log("test_acc", self.acc)
        pred = torch.argmax(output, dim=-1)
        if batch_idx == 0:
            self.logger.experiment[0].add_images("pred_img", data, 0)
            self.logger.experiment[0].add_text(
                "test_pred",
                ", ".join(str(i) for i in pred.cpu().detach().numpy()), 0)

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(),
                                lr=self.learning_rate,
                                amsgrad=True)
