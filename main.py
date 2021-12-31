import math
import os
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from pytorch_lightning.callbacks import DeviceStatsMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer
import torch

from data.datamodule import MusicDataModule
from model.model import MusicModel


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig = None) -> None:
    model = MusicModel(d_model=cfg.model.d_model,
                       data_len=cfg.model.data_len,
                       dropout=cfg.model.dropout,
                       ff=cfg.model.ff,
                       lr=cfg.train.lr,
                       nhead=cfg.model.nhead,
                       num_layers=cfg.model.num_layers,
                       num_token=cfg.model.num_token,
                       segments=cfg.model.segments)
    datamodule = MusicDataModule(cfg)
    callbacks = []
    if cfg.train.checkpoint:
        callbacks.append(
            ModelCheckpoint(
                dirpath=to_absolute_path("checkpoints"),
                filename="epoch={epoch}-val_loss={val/loss:.3f}",
                monitor="val/loss",
                save_top_k=3,
                mode="min",
                auto_insert_metric_name=False,
                save_weights_only=True,
            ))
    if cfg.train.monitor:
        callbacks.append(DeviceStatsMonitor())
    if cfg.train.early_stopping:
        callbacks.append(EarlyStopping(monitor="val/loss", mode="min"))
    logger = WandbLogger(project="music-model", save_dir=to_absolute_path("."))
    devices = "auto" if cfg.train.gpus == -1 else cfg.train.gpus
    max_time = None if cfg.train.max_time == "" else cfg.train.max_time
    trainer = Trainer(
        accelerator="auto",
        accumulate_grad_batches=cfg.train.acc,
        auto_lr_find=cfg.train.auto_lr,
        auto_scale_batch_size=cfg.train.auto_batch,
        callbacks=callbacks,
        detect_anomaly=True,
        devices=devices,
        fast_dev_run=cfg.train.fast_dev_run,
        limit_train_batches=cfg.train.limit_batches,
        limit_val_batches=cfg.train.limit_batches,
        limit_test_batches=cfg.train.limit_batches,
        log_every_n_steps=1,
        logger=[logger],
        max_time=max_time,
        num_sanity_val_steps=2,
        precision=16,
    )

    try:
        trainer.tune(model=model,
                     datamodule=datamodule,
                     lr_find_kwargs={"max_lr": 0.01})
        trainer.fit(model=model, datamodule=datamodule)
        trainer.test(model=model, datamodule=datamodule)

        os.makedirs(to_absolute_path("onnx"), exist_ok=True)
        model.to_onnx(file_path=to_absolute_path(Path("onnx", "model.onnx")),
                      export_params=True)
    except RuntimeError:
        for parameter in model.parameters():
            if parameter.grad is not None:
                del parameter.grad
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        try:
            trainer.validate(model=model, datamodule=datamodule)
        except RuntimeError:
            logger.log_metrics(
                metrics={"val/loss": math.log(cfg.model.num_tokens)})
            trainer.teardown()


if __name__ == "__main__":
    main()
