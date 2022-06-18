"""
    The main file of the project.
    This file initializes the model and data, and then runs the training.
"""
import hydra
from hydra.utils import to_absolute_path
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


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig = None) -> None:
    """
        The main function.
        To run in another file, initialize a hydra config and call main(cfg).
    """

    # Initialize CustomConfig
    custom_cfg = CustomConfig(cfg)
    datamodule = MusicDataModule(cfg)
    devices = "auto" if custom_cfg.gpus == -1 else custom_cfg.gpus

    batch_size = custom_cfg.batch_size
    if custom_cfg.auto_batch:
        batch_model = MusicModel(custom_cfg)
        batch_trainer = Trainer(
            accelerator="auto",
            accumulate_grad_batches=2,
            detect_anomaly=True,
            devices=devices,
            precision=16,
        )
        try:
            batch_size = batch_trainer.tuner.scale_batch_size(
                model=batch_model, datamodule=datamodule
            )
            datamodule.batch_size = batch_size
        except RuntimeError:
            return
        del batch_model, batch_trainer

    if custom_cfg.effective_batch_size > 0:
        accumulate = custom_cfg.effective_batch_size // batch_size
        accumulate = max(accumulate, 1)
    else:
        accumulate = custom_cfg.acc

    if custom_cfg.auto_lr:
        lr_model = MusicModel(custom_cfg)
        lr_trainer = Trainer(
            accelerator="auto",
            accumulate_grad_batches=accumulate,
            detect_anomaly=True,
            devices=devices,
            precision=16,
        )
        lr_finder = lr_trainer.tuner.lr_find(
            model=lr_model, datamodule=datamodule, max_lr=0.01
        )
        custom_cfg.learning_rate = lr_finder.suggestion()
        del lr_model, lr_trainer

    model = MusicModel(custom_cfg)
    callbacks = []
    if custom_cfg.checkpoint:
        callbacks.append(
            ModelCheckpoint(
                dirpath=to_absolute_path("checkpoints"),
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
    logger = WandbLogger(project="music-model", save_dir=to_absolute_path("."))
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
        max_time=max_time,
        num_sanity_val_steps=2,
        precision=16,
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
