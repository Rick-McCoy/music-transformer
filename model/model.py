from pytorch_lightning import LightningModule
import torch
from torch import Tensor
from torchmetrics import Accuracy

from model.loss import SimpleLoss
from model.transformer import Transformer


class MusicModel(LightningModule):
    def __init__(self, d_model: int, data_len: int, dropout: float, ff: int,
                 lr: float, nhead: int, num_layers: int, num_token: int,
                 segments: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = lr
        self.transformer = Transformer(d_model=d_model,
                                       data_len=data_len,
                                       dropout=dropout,
                                       ff=ff,
                                       nhead=nhead,
                                       num_layers=num_layers,
                                       num_token=num_token,
                                       segments=segments)
        self.loss = SimpleLoss()
        self.acc = Accuracy(top_k=1, ignore_index=0)
        self.example_input_array = torch.zeros(1, data_len, dtype=torch.int64)

    def forward(self, data: Tensor, *args, **kwargs) -> Tensor:
        return self.transformer(data)

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
        return torch.optim.Adam(params=self.parameters(),
                                lr=self.learning_rate,
                                amsgrad=True)
