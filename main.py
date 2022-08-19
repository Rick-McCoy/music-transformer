"""
    The main file of the project.
    This file initializes the model and data, and then runs the training.
"""
import gc
import math
import traceback

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.trainer import Trainer

from config.config import CustomConfig
from data.datamodule import MusicDataModule
from model.model import MusicModel


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
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
            if custom_cfg.effective_batch_size > 0:
                max_trials = round(math.log2(custom_cfg.effective_batch_size / 2))
            else:
                max_trials = 25
            batch_size = batch_trainer.tuner.scale_batch_size(
                model=batch_model,
                datamodule=datamodule,
                steps_per_trial=10,
                max_trials=max_trials,
            )
            assert batch_size is not None
            datamodule.batch_size = batch_size
        except RuntimeError:
            return
        del batch_model, batch_trainer
        torch.cuda.empty_cache()
        gc.collect()

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
        assert lr_finder is not None
        learning_rate = lr_finder.suggestion()
        assert isinstance(learning_rate, float)
        custom_cfg.learning_rate = learning_rate
        print(f"Learning rate: {custom_cfg.learning_rate}")
        del lr_model, lr_trainer
        torch.cuda.empty_cache()
        gc.collect()

    model = MusicModel(
        learning_rate=custom_cfg.learning_rate,
        data_len=custom_cfg.data_len,
        d_model=custom_cfg.d_model,
        dropout=custom_cfg.dropout,
        feed_forward=custom_cfg.feed_forward,
        nhead=custom_cfg.nhead,
        num_layers=custom_cfg.num_layers,
        num_tokens=custom_cfg.num_tokens,
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
        callbacks.append(EarlyStopping(monitor="val/loss", mode="min", patience=10))

    if custom_cfg.profile:
        profiler = AdvancedProfiler(dirpath=custom_cfg.profile_dir, filename="perf_logs")
    else:
        profiler = None

    custom_cfg.log_dir.mkdir(parents=True, exist_ok=True)
    if custom_cfg.wandb:
        logger = WandbLogger(save_dir=str(custom_cfg.log_dir))
    else:
        logger = None
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
        logger=[logger] if logger is not None else False,
        max_epochs=custom_cfg.max_epochs,
        max_time=max_time,
        num_sanity_val_steps=2,
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
