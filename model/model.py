from typing import Dict, List, Tuple
import torch
from torch import Tensor
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

from model.accuracy import SimpleAccuracy
from model.loss import SimpleLoss
from model.transformer import Transformer


class MusicModel(LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.learning_rate = cfg.train.lr
        self.transformer = Transformer(cfg)
        self.loss = SimpleLoss(cfg)
        self.acc = SimpleAccuracy(cfg)
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

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor],
                      *args, **kwargs) -> Tensor:
        tick, pitch, program, velocity = batch
        tick_out, pitch_out, program_out, velocity_out = self(
            tick[:, :-1], pitch[:, :-1], program[:, :-1], velocity[:, :-1])
        tick_loss = self.loss(tick_out, tick[:, 1:])
        pitch_loss = self.loss(pitch_out, pitch[:, 1:])
        program_loss = self.loss(program_out, program[:, 1:])
        velocity_loss = self.loss(velocity_out, velocity[:, 1:])
        loss = tick_loss + pitch_loss + program_loss + velocity_loss
        tick_acc = self.acc(tick_out, tick[:, 1:])
        pitch_acc = self.acc(pitch_out, pitch[:, 1:])
        program_acc = self.acc(program_out, program[:, 1:])
        velocity_acc = self.acc(velocity_out, velocity[:, 1:])
        acc = tick_acc + pitch_acc + program_acc + velocity_acc
        self.log("train_tick_loss", tick_loss)
        self.log("train_tick_acc", tick_acc)
        self.log("train_pitch_loss", pitch_loss)
        self.log("train_pitch_acc", pitch_acc)
        self.log("train_program_loss", program_loss)
        self.log("train_program_acc", program_acc)
        self.log("train_velocity_loss", velocity_loss)
        self.log("train_velocity_acc", velocity_acc)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], *args,
                        **kwargs) -> Tensor:
        tick, pitch, program, velocity = batch
        tick_out, pitch_out, program_out, velocity_out = self.transformer(
            tick[:, :-1], pitch[:, :-1], program[:, :-1], velocity[:, :-1])
        tick_loss = self.loss(tick_out, tick[:, 1:])
        pitch_loss = self.loss(pitch_out, pitch[:, 1:])
        program_loss = self.loss(program_out, program[:, 1:])
        velocity_loss = self.loss(velocity_out, velocity[:, 1:])
        loss = tick_loss + pitch_loss + program_loss + velocity_loss
        tick_acc = self.acc(tick_out, tick[:, 1:])
        pitch_acc = self.acc(pitch_out, pitch[:, 1:])
        program_acc = self.acc(program_out, program[:, 1:])
        velocity_acc = self.acc(velocity_out, velocity[:, 1:])
        acc = tick_acc + pitch_acc + program_acc + velocity_acc
        self.log("val_tick_loss", tick_loss)
        self.log("val_tick_acc", tick_acc)
        self.log("val_pitch_loss", pitch_loss)
        self.log("val_pitch_acc", pitch_acc)
        self.log("val_program_loss", program_loss)
        self.log("val_program_acc", program_acc)
        self.log("val_velocity_loss", velocity_loss)
        self.log("val_velocity_acc", velocity_acc)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        tick_pred = torch.argmax(tick_out, dim=-1)
        pitch_pred = torch.argmax(pitch_out, dim=-1)
        program_pred = torch.argmax(program_out, dim=-1)
        velocity_pred = torch.argmax(velocity_out, dim=-1)
        return {
            'tick': tick[:, 1:],
            'tick_pred': tick_pred,
            'pitch': pitch[:, 1:],
            'pitch_pred': pitch_pred,
            'program': program[:, 1:],
            'program_pred': program_pred,
            'velocity': velocity[:, 1:],
            'velocity_pred': velocity_pred,
        }

    def validation_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> None:
        ticks = torch.cat([output['tick'] for output in outputs], dim=0)
        tick_preds = torch.cat([output['tick_pred'] for output in outputs],
                               dim=0)
        self.logger.experiment[0].add_pr_curve("tick", ticks, tick_preds,
                                               self.current_epoch)
        pitches = torch.cat([output['pitch'] for output in outputs], dim=0)
        pitch_preds = torch.cat([output['pitch_pred'] for output in outputs],
                                dim=0)
        self.logger.experiment[0].add_pr_curve("pitch", pitches, pitch_preds,
                                               self.current_epoch)
        programs = torch.cat([output['program'] for output in outputs], dim=0)
        program_preds = torch.cat(
            [output['program_pred'] for output in outputs], dim=0)
        self.logger.experiment[0].add_pr_curve("program", programs,
                                               program_preds,
                                               self.current_epoch)
        velocities = torch.cat([output['velocity'] for output in outputs],
                               dim=0)
        velocity_preds = torch.cat(
            [output['velocity_pred'] for output in outputs], dim=0)
        self.logger.experiment[0].add_pr_curve("velocity", velocities,
                                               velocity_preds,
                                               self.current_epoch)

    def test_step(self, batch: Tuple[Tensor, Tensor], *args,
                  **kwargs) -> Tensor:
        tick, pitch, program, velocity = batch
        tick_out, pitch_out, program_out, velocity_out = self.transformer(
            tick[:, :-1], pitch[:, :-1], program[:, :-1], velocity[:, :-1])
        tick_loss = self.loss(tick_out, tick[:, 1:])
        pitch_loss = self.loss(pitch_out, pitch[:, 1:])
        program_loss = self.loss(program_out, program[:, 1:])
        velocity_loss = self.loss(velocity_out, velocity[:, 1:])
        loss = tick_loss + pitch_loss + program_loss + velocity_loss
        tick_acc = self.acc(tick_out, tick[:, 1:])
        pitch_acc = self.acc(pitch_out, pitch[:, 1:])
        program_acc = self.acc(program_out, program[:, 1:])
        velocity_acc = self.acc(velocity_out, velocity[:, 1:])
        acc = tick_acc + pitch_acc + program_acc + velocity_acc
        self.log("test_tick_loss", tick_loss)
        self.log("test_tick_acc", tick_acc)
        self.log("test_pitch_loss", pitch_loss)
        self.log("test_pitch_acc", pitch_acc)
        self.log("test_program_loss", program_loss)
        self.log("test_program_acc", program_acc)
        self.log("test_velocity_loss", velocity_loss)
        self.log("test_velocity_acc", velocity_acc)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        tick_pred = torch.argmax(tick_out, dim=-1)
        pitch_pred = torch.argmax(pitch_out, dim=-1)
        program_pred = torch.argmax(program_out, dim=-1)
        velocity_pred = torch.argmax(velocity_out, dim=-1)
        return {
            'tick': tick[:, 1:],
            'tick_pred': tick_pred,
            'pitch': pitch[:, 1:],
            'pitch_pred': pitch_pred,
            'program': program[:, 1:],
            'program_pred': program_pred,
            'velocity': velocity[:, 1:],
            'velocity_pred': velocity_pred,
        }

    def test_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> None:
        ticks = torch.cat([output['tick'] for output in outputs], dim=0)
        tick_preds = torch.cat([output['tick_pred'] for output in outputs],
                               dim=0)
        self.logger.experiment[0].add_pr_curve("tick", ticks, tick_preds,
                                               self.current_epoch)
        pitches = torch.cat([output['pitch'] for output in outputs], dim=0)
        pitch_preds = torch.cat([output['pitch_pred'] for output in outputs],
                                dim=0)
        self.logger.experiment[0].add_pr_curve("pitch", pitches, pitch_preds,
                                               self.current_epoch)
        programs = torch.cat([output['program'] for output in outputs], dim=0)
        program_preds = torch.cat(
            [output['program_pred'] for output in outputs], dim=0)
        self.logger.experiment[0].add_pr_curve("program", programs,
                                               program_preds,
                                               self.current_epoch)
        velocities = torch.cat([output['velocity'] for output in outputs],
                               dim=0)
        velocity_preds = torch.cat(
            [output['velocity_pred'] for output in outputs], dim=0)
        self.logger.experiment[0].add_pr_curve("velocity", velocities,
                                               velocity_preds,
                                               self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(),
                                lr=self.learning_rate,
                                amsgrad=True)
