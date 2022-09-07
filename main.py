"""
    The main file of the project.
    This file initializes the model and data, and then runs the training.
"""
import gc
import math
import traceback
import warnings
from typing import Tuple

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.profiler import PyTorchProfiler
from pytorch_lightning.trainer import Trainer

from config.config import CustomConfig
from data.datamodule import MusicDataModule
from model.model import MusicModel


def get_tune_model(cfg: CustomConfig) -> Tuple[MusicModel, Trainer]:
    devices = "auto" if cfg.gpus == -1 else cfg.gpus
    batch_model = MusicModel(
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        data_len=cfg.data_len,
        d_model=cfg.d_model,
        dropout=cfg.dropout,
        feed_forward=cfg.feed_forward,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
    )
    batch_trainer = Trainer(
        accelerator="auto",
        accumulate_grad_batches=cfg.acc,
        detect_anomaly=True,
        devices=devices,
        logger=False,
        max_epochs=cfg.max_epochs,
        precision=16,
    )

    return batch_model, batch_trainer


def tune_batch_size(cfg: CustomConfig) -> int:
    datamodule = MusicDataModule(cfg=cfg)
    cfg.acc = cfg.acc if cfg.acc > 1 else 8
    batch_model, batch_trainer = get_tune_model(cfg)
    if cfg.effective_batch_size > 0:
        max_trials = round(math.log2(cfg.effective_batch_size / 2))
    else:
        max_trials = 25
    batch_size = batch_trainer.tuner.scale_batch_size(
        model=batch_model,
        datamodule=datamodule,
        steps_per_trial=10,
        max_trials=max_trials,
    )
    assert batch_size is not None
    del batch_model, batch_trainer, datamodule
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()

    return batch_size


def tune_learning_rate(cfg: CustomConfig) -> float:
    datamodule = MusicDataModule(cfg=cfg)
    cfg.acc = cfg.acc if cfg.acc > 1 else 8
    lr_model, lr_trainer = get_tune_model(cfg)
    lr_finder = lr_trainer.tuner.lr_find(
        model=lr_model,
        datamodule=datamodule,
    )
    assert lr_finder is not None
    fig = lr_finder.plot(suggest=True)
    assert fig is not None
    fig.savefig("lr_finder.png")
    learning_rate = lr_finder.suggestion()
    assert isinstance(learning_rate, float)
    del lr_model, lr_trainer, datamodule
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()

    return learning_rate


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    The main function.
    To run in another file, initialize a hydra config and call main(cfg).
    """

    warnings.filterwarnings(
        "ignore",
        message="None of the inputs have requires_grad=True. Gradients will be None",
        category=UserWarning,
    )

    # Initialize CustomConfig
    custom_cfg = CustomConfig(cfg)
    devices = "auto" if custom_cfg.gpus == -1 else custom_cfg.gpus

    if custom_cfg.auto_batch:
        try:
            custom_cfg.batch_size = tune_batch_size(custom_cfg)
        except RuntimeError:
            print("RuntimeError: Batch size tuning failed.")
            traceback.print_exc()
            return

    if custom_cfg.effective_batch_size > 0:
        custom_cfg.acc = custom_cfg.effective_batch_size // custom_cfg.batch_size

    if custom_cfg.auto_lr:
        try:
            custom_cfg.learning_rate = tune_learning_rate(custom_cfg)
            print(f"Learning rate: {custom_cfg.learning_rate}")
        except RuntimeError:
            print("RuntimeError: Learning rate tuning failed.")
            traceback.print_exc()
            return
    else:
        custom_cfg.learning_rate /= custom_cfg.acc

    model = MusicModel(
        learning_rate=custom_cfg.learning_rate,
        weight_decay=custom_cfg.weight_decay,
        data_len=custom_cfg.data_len,
        d_model=custom_cfg.d_model,
        dropout=custom_cfg.dropout,
        feed_forward=custom_cfg.feed_forward,
        nhead=custom_cfg.nhead,
        num_layers=custom_cfg.num_layers,
    )
    datamodule = MusicDataModule(cfg=custom_cfg)

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
        callbacks.append(EarlyStopping(monitor="val/loss", mode="min", patience=10))

    if custom_cfg.profile:
        profiler = PyTorchProfiler(
            dirpath=custom_cfg.profile_dir, filename="perf_logs", export_to_chrome=True
        )
    else:
        profiler = None

    custom_cfg.log_dir.mkdir(parents=True, exist_ok=True)
    if custom_cfg.wandb:
        logger = WandbLogger(save_dir=str(custom_cfg.log_dir), project="music-transformer")
        logger.log_hyperparams({"effective_batch_size": custom_cfg.effective_batch_size})
        logger.watch(model)
    else:
        logger = None
    max_time = None if custom_cfg.max_time == "" else custom_cfg.max_time

    trainer = Trainer(
        accelerator="auto",
        accumulate_grad_batches=custom_cfg.acc,
        callbacks=callbacks,
        detect_anomaly=True,
        devices=devices,
        fast_dev_run=custom_cfg.fast_dev_run,
        limit_train_batches=custom_cfg.limit_batches,
        limit_val_batches=custom_cfg.limit_batches,
        limit_test_batches=custom_cfg.limit_batches,
        log_every_n_steps=1,
        logger=[logger] if logger is not None else False,
        max_epochs=custom_cfg.max_epochs,
        max_time=max_time,
        num_sanity_val_steps=10,
        precision=16,
        profiler=profiler,
    )

    error_tuple = (RuntimeError,) if custom_cfg.ignore_runtime_error else tuple()
    try:
        trainer.fit(model=model, datamodule=datamodule)
    except error_tuple as exception:
        print(f"Error: {exception.__class__.__name__}. Traceback: {traceback.format_exc()}")
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
