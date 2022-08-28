"""
    Defines MusicModel class.
"""

from typing import List, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor

import wandb
from config.config import NUM_TOKEN
from model.accuracy import SimpleAccuracy
from model.loss import CrossEntropy
from model.simplify import SimplifyClass, SimplifyScore
from model.transformer import Transformer
from wandb import plot as wandb_plot


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

    def forward(self, *args, **kwargs) -> Tensor:
        data = args[0]
        return self.transformer(data)

    def step_template(self, *args, **_kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        batch = args[0]
        output = self(batch[:, :-1])
        loss = self.loss(output, batch[:, 1:])
        acc = self.acc(output, batch[:, 1:])
        self.log("train/loss", loss)
        self.log("train/acc", acc)
        return loss, output, batch[:, 1:]

    def training_step(self, *args, **kwargs) -> Tensor:
        loss, *_ = self.step_template(*args, **kwargs)
        return loss

    def validation_step(self, *args, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        return self.step_template(*args, **kwargs)

    def test_step(self, *args, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        return self.step_template(*args, **kwargs)

    def epoch_end_template(self, outputs: List[Tuple[Tensor, Tensor, Tensor]], mode: str) -> None:
        _, output_list, target_list = zip(*outputs)
        score = torch.cat(output_list, dim=0).permute([0, 2, 1]).flatten(0, 1)
        target = torch.cat(target_list, dim=0).flatten()
        y_true = self.simplify_class(target).detach().cpu().numpy()
        y_probas = self.simplify_score(score).detach().cpu().numpy()
        assert mode in ["val", "test"], f"Unknown mode {mode}"
        if self.logger is not None:
            wandb.log(
                {
                    f"{mode}/pr_curve": wandb_plot.pr_curve(
                        y_true=y_true[:10000],
                        y_probas=y_probas[:10000],
                        labels=self.class_names,
                    )
                }
            )
            wandb.log(
                {
                    f"{mode}/roc_curve": wandb_plot.roc_curve(
                        y_true=y_true[:10000],
                        y_probas=y_probas[:10000],
                        labels=self.class_names,
                    )
                }
            )
            wandb.log(
                {
                    f"{mode}/conf_mat": wandb_plot.confusion_matrix(
                        probs=y_probas,
                        y_true=y_true,
                        class_names=self.class_names,
                    )
                }
            )

    def validation_epoch_end(self, outputs: List[Tuple[Tensor, Tensor, Tensor]]) -> None:
        self.epoch_end_template(outputs, "val")

    def test_epoch_end(self, outputs: List[Tuple[Tensor, Tensor, Tensor]]) -> None:
        self.epoch_end_template(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            amsgrad=True,
        )
