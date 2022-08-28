"""
    Defines MusicModel class.
"""

from typing import List, Tuple

import torch
import wandb
from pytorch_lightning import LightningModule
from torch import Tensor
from wandb import plot as wandb_plot

from config.config import NUM_TOKEN
from model.accuracy import SimpleAccuracy
from model.loss import CrossEntropy
from model.simplify import SimplifyClass, SimplifyScore
from model.transformer import Transformer


class MusicModel(LightningModule):
    """
    The MusicModel class.
    This implements various Lightningmodule methods.
    """

    def __init__(
        self,
        learning_rate: float,
        weight_decay: float,
        d_model: int,
        data_len: int,
        dropout: float,
        feed_forward: int,
        nhead: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.transformer = Transformer(
            d_model=d_model,
            data_len=data_len,
            dropout=dropout,
            feed_forward=feed_forward,
            nhead=nhead,
            num_layers=num_layers,
            num_token=NUM_TOKEN,
        )
        self.loss = CrossEntropy()
        self.acc = SimpleAccuracy()
        self.simplify_class = SimplifyClass()
        self.simplify_score = SimplifyScore()
        self.example_input_array = torch.zeros(1, data_len, dtype=torch.int64)
        self.class_names = ["Special", "Program", "Drum", "Note", "Tick"]

    def forward(self, data: Tensor) -> Tensor:  # pylint: disable=arguments-differ
        return self.transformer(data)

    def training_step(self, *args, **kwargs) -> Tensor:  # pylint: disable=unused-argument
        batch = args[0]
        output = self(batch[:, :-1])
        loss = self.loss(output, batch[:, 1:])
        acc = self.acc(output, batch[:, 1:])
        self.log("train/loss", loss)
        self.log("train/acc", acc)
        return loss

    def validation_step(
        self, *args, **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor]:  # pylint: disable=unused-argument
        batch = args[0]
        output = self(batch[:, :-1])
        loss = self.loss(output, batch[:, 1:])
        acc = self.acc(output, batch[:, 1:])
        self.log("val/loss", loss)
        self.log("val/acc", acc)
        return loss, output, batch[:, 1:]

    def validation_epoch_end(self, outputs: List[Tuple[Tensor, Tensor, Tensor]]) -> None:
        _, output_list, target_list = zip(*outputs)
        score = torch.cat(output_list, dim=0).permute([0, 2, 1]).flatten(0, 1)
        target = torch.cat(target_list, dim=0).flatten()
        y_true = self.simplify_class(target).detach().cpu().numpy()
        y_probas = self.simplify_score(score).detach().cpu().numpy()
        if self.logger is not None:
            wandb.log(
                {
                    "val/pr_curve": wandb_plot.pr_curve(
                        y_true=y_true[:10000],
                        y_probas=y_probas[:10000],
                        labels=self.class_names,
                    )
                }
            )
            wandb.log(
                {
                    "val/roc_curve": wandb_plot.roc_curve(
                        y_true=y_true[:10000],
                        y_probas=y_probas[:10000],
                        labels=self.class_names,
                    )
                }
            )
            wandb.log(
                {
                    "val/conf_mat": wandb_plot.confusion_matrix(
                        probs=y_probas,
                        y_true=y_true,
                        class_names=self.class_names,
                    )
                }
            )

    def test_step(
        self, *args, **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor]:  # pylint: disable=unused-argument
        batch = args[0]
        output = self(batch[:, :-1])
        loss = self.loss(output, batch[:, 1:])
        acc = self.acc(output, batch[:, 1:])
        self.log("test/loss", loss)
        self.log("test/acc", acc)
        return loss, output, batch[:, 1:]

    def test_epoch_end(self, outputs: List[Tuple[Tensor, Tensor, Tensor]]) -> None:
        _, output_list, target_list = zip(*outputs)
        score = torch.cat(output_list, dim=0).permute([0, 2, 1]).flatten(0, 1)
        target = torch.cat(target_list, dim=0).flatten()
        y_true = self.simplify_class(target).detach().cpu().numpy()
        y_probas = self.simplify_score(score).detach().cpu().numpy()
        if self.logger is not None:
            wandb.log(
                {
                    "test/pr_curve": wandb_plot.pr_curve(
                        y_true=y_true[:10000],
                        y_probas=y_probas[:10000],
                        labels=self.class_names,
                    )
                }
            )
            wandb.log(
                {
                    "test/roc_curve": wandb_plot.roc_curve(
                        y_true=y_true[:10000],
                        y_probas=y_probas[:10000],
                        labels=self.class_names,
                    )
                }
            )
            wandb.log(
                {
                    "test/conf_mat": wandb_plot.confusion_matrix(
                        probs=y_probas,
                        y_true=y_true,
                        class_names=self.class_names,
                    )
                }
            )

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            amsgrad=True,
        )
