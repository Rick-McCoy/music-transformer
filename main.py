"""
    The main file of the project.
    This file initializes the model and data, and then runs the training.
"""
import math

import hydra
from omegaconf import DictConfig
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer

from config.config import CustomConfig
from data.datamodule import MusicDataModule
from model.model import MusicModel


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig = None) -> None:
    """
    The main function.
    To run in another file, initialize a hydra config and call main(cfg).
    """

    # Initialize CustomConfig
    custom_cfg = CustomConfig(cfg)
    datamodule = MusicDataModule(custom_cfg)
    devices = "auto" if custom_cfg.gpus == -1 else custom_cfg.gpus

    batch_size = custom_cfg.batch_size
    if custom_cfg.auto_batch:
        batch_model = MusicModel(
            learning_rate=custom_cfg.learning_rate,
            data_len=custom_cfg.data_len,
            d_model=custom_cfg.d_model,
            dropout=custom_cfg.dropout,
            feed_forward=custom_cfg.feed_forward,
            nhead=custom_cfg.nhead,
            num_layers=custom_cfg.num_layers,
            num_tokens=custom_cfg.num_tokens,
            segments=custom_cfg.segments,
            is_training=False,
        )
        batch_trainer = Trainer(
            accelerator="auto",
            accumulate_grad_batches=2,
            detect_anomaly=True,
            devices=devices,
            max_epochs=custom_cfg.max_epochs,
            precision=16,
        )
        try:
            max_trials = round(math.log2(custom_cfg.effective_batch_size / 2))
            batch_size = batch_trainer.tuner.scale_batch_size(
                model=batch_model,
                datamodule=datamodule,
                steps_per_trial=10,
                max_trials=max_trials,
            )
            datamodule.batch_size = batch_size
        except RuntimeError:
            return
        del batch_model, batch_trainer

    if custom_cfg.effective_batch_size > 0:
        accumulate = custom_cfg.effective_batch_size // batch_size
    else:
        accumulate = custom_cfg.acc

    if custom_cfg.auto_lr:
        lr_model = MusicModel(
            learning_rate=custom_cfg.learning_rate,
            data_len=custom_cfg.data_len,
            d_model=custom_cfg.d_model,
            dropout=custom_cfg.dropout,
            feed_forward=custom_cfg.feed_forward,
            nhead=custom_cfg.nhead,
            num_layers=custom_cfg.num_layers,
            num_tokens=custom_cfg.num_tokens,
            segments=custom_cfg.segments,
            is_training=False,
        )
        lr_trainer = Trainer(
            accelerator="auto",
            accumulate_grad_batches=accumulate,
            detect_anomaly=True,
            devices=devices,
            max_epochs=custom_cfg.max_epochs,
            precision=16,
        )
        lr_finder = lr_trainer.tuner.lr_find(
            model=lr_model,
            datamodule=datamodule,
            max_lr=0.01,
        )
        custom_cfg.learning_rate = lr_finder.suggestion()
        print(f"Learning rate: {custom_cfg.learning_rate}")
        del lr_model, lr_trainer

    model = MusicModel(
        learning_rate=custom_cfg.learning_rate,
        data_len=custom_cfg.data_len,
        d_model=custom_cfg.d_model,
        dropout=custom_cfg.dropout,
        feed_forward=custom_cfg.feed_forward,
        nhead=custom_cfg.nhead,
        num_layers=custom_cfg.num_layers,
        num_tokens=custom_cfg.num_tokens,
        segments=custom_cfg.segments,
    )

    callbacks = []
    if custom_cfg.checkpoint:
        callbacks.append(
            ModelCheckpoint(
                dirpath=custom_cfg.checkpoint_dir,
                filename="epoch={epoch}-val_loss={val/loss:.3f}",
                monitor="val/loss",
                save_top_k=1,
                mode="min",
                auto_insert_metric_name=False,
                save_weights_only=True,
            )
        )
    if custom_cfg.monitor:
        callbacks.append(DeviceStatsMonitor())
    if custom_cfg.early_stop:
        callbacks.append(EarlyStopping(monitor="val/loss", mode="min"))

    custom_cfg.log_dir.mkdir(parents=True, exist_ok=True)
    logger = WandbLogger(name="music-model", save_dir=str(custom_cfg.log_dir))
    max_time = None if custom_cfg.max_time == "" else custom_cfg.max_time

    trainer = Trainer(
        accelerator="auto",
        accumulate_grad_batches=accumulate,
        callbacks=callbacks,
        detect_anomaly=True,
        devices=devices,
        fast_dev_run=custom_cfg.fast_dev_run,
        limit_train_batches=custom_cfg.limit_batches,
        limit_val_batches=custom_cfg.limit_batches,
        limit_test_batches=custom_cfg.limit_batches,
        log_every_n_steps=1,
        logger=[logger],
        max_epochs=custom_cfg.max_epochs,
        max_time=max_time,
        num_sanity_val_steps=2,
        precision=16,
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
