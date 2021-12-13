from typing import List

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
        self.pitch_acc = Accuracy(top_k=1)
        self.program_acc = Accuracy(top_k=1)
        self.tick_acc = Accuracy(top_k=1)
        self.velocity_acc = Accuracy(top_k=1)
        self.example_input_array = (torch.zeros(cfg.train.batch_size,
                                                cfg.model.data_len,
                                                dtype=torch.int64),
                                    torch.zeros(cfg.train.batch_size,
                                                cfg.model.data_len,
                                                dtype=torch.int64),
                                    torch.zeros(cfg.train.batch_size,
                                                cfg.model.data_len,
                                                dtype=torch.int64),
                                    torch.zeros(cfg.train.batch_size,
                                                cfg.model.data_len,
                                                dtype=torch.int64))

    def forward(self, tick: Tensor, pitch: Tensor, program: Tensor,
                velocity: Tensor, *args, **kwargs) -> Tensor:
        return self.transformer(tick, pitch, program, velocity)

    def training_step(self, batch: List[Tensor], *args, **kwargs) -> Tensor:
        tick, pitch, program, velocity = batch
        tick_out, pitch_out, program_out, velocity_out = self(
            tick[:, :-1], pitch[:, :-1], program[:, :-1], velocity[:, :-1])
        tick_loss = self.loss(tick_out, tick[:, 1:])
        pitch_loss = self.loss(pitch_out, pitch[:, 1:])
        program_loss = self.loss(program_out, program[:, 1:])
        velocity_loss = self.loss(velocity_out, velocity[:, 1:])
        loss = tick_loss + pitch_loss + program_loss + velocity_loss
        tick_acc = self.tick_acc(tick_out, tick[:, 1:])
        pitch_acc = self.pitch_acc(pitch_out, pitch[:, 1:])
        program_acc = self.program_acc(program_out, program[:, 1:])
        velocity_acc = self.velocity_acc(velocity_out, velocity[:, 1:])
        acc = torch.mean(
            torch.stack([tick_acc, pitch_acc, program_acc, velocity_acc]))
        self.log("train_tick_loss", tick_loss)
        self.log("train_tick_acc", self.tick_acc)
        self.log("train_pitch_loss", pitch_loss)
        self.log("train_pitch_acc", self.pitch_acc)
        self.log("train_program_loss", program_loss)
        self.log("train_program_acc", self.program_acc)
        self.log("train_velocity_loss", velocity_loss)
        self.log("train_velocity_acc", self.velocity_acc)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch: List[Tensor], *args, **kwargs) -> Tensor:
        tick, pitch, program, velocity = batch
        tick_out, pitch_out, program_out, velocity_out = self(
            tick[:, :-1], pitch[:, :-1], program[:, :-1], velocity[:, :-1])
        tick_loss = self.loss(tick_out, tick[:, 1:])
        pitch_loss = self.loss(pitch_out, pitch[:, 1:])
        program_loss = self.loss(program_out, program[:, 1:])
        velocity_loss = self.loss(velocity_out, velocity[:, 1:])
        loss = tick_loss + pitch_loss + program_loss + velocity_loss
        tick_acc = self.tick_acc(tick_out, tick[:, 1:])
        pitch_acc = self.pitch_acc(pitch_out, pitch[:, 1:])
        program_acc = self.program_acc(program_out, program[:, 1:])
        velocity_acc = self.velocity_acc(velocity_out, velocity[:, 1:])
        acc = torch.mean(
            torch.stack([tick_acc, pitch_acc, program_acc, velocity_acc]))
        self.log("val_tick_loss", tick_loss)
        self.log("val_tick_acc", self.tick_acc)
        self.log("val_pitch_loss", pitch_loss)
        self.log("val_pitch_acc", self.pitch_acc)
        self.log("val_program_loss", program_loss)
        self.log("val_program_acc", self.program_acc)
        self.log("val_velocity_loss", velocity_loss)
        self.log("val_velocity_acc", self.velocity_acc)
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def test_step(self, batch: List[Tensor], *args, **kwargs) -> Tensor:
        tick, pitch, program, velocity = batch
        tick_out, pitch_out, program_out, velocity_out = self(
            tick[:, :-1], pitch[:, :-1], program[:, :-1], velocity[:, :-1])
        tick_loss = self.loss(tick_out, tick[:, 1:])
        pitch_loss = self.loss(pitch_out, pitch[:, 1:])
        program_loss = self.loss(program_out, program[:, 1:])
        velocity_loss = self.loss(velocity_out, velocity[:, 1:])
        loss = tick_loss + pitch_loss + program_loss + velocity_loss
        tick_acc = self.tick_acc(tick_out, tick[:, 1:])
        pitch_acc = self.pitch_acc(pitch_out, pitch[:, 1:])
        program_acc = self.program_acc(program_out, program[:, 1:])
        velocity_acc = self.velocity_acc(velocity_out, velocity[:, 1:])
        acc = torch.mean(
            torch.stack([tick_acc, pitch_acc, program_acc, velocity_acc]))
        self.log("test_tick_loss", tick_loss)
        self.log("test_tick_acc", self.tick_acc)
        self.log("test_pitch_loss", pitch_loss)
        self.log("test_pitch_acc", self.pitch_acc)
        self.log("test_program_loss", program_loss)
        self.log("test_program_acc", self.program_acc)
        self.log("test_velocity_loss", velocity_loss)
        self.log("test_velocity_acc", self.velocity_acc)
        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(),
                                lr=self.learning_rate,
                                amsgrad=True)
