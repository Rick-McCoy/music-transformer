import os

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from pytorch_lightning.callbacks import DeviceStatsMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.trainer import Trainer

from data.datamodule import SimpleDataModule
from model.model import SimpleModel


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig = None) -> None:
    model = SimpleModel(cfg)
    datamodule = SimpleDataModule(cfg)
    callbacks = []
    if cfg.train.checkpoint:
        callbacks.append(
            ModelCheckpoint(
                dirpath=to_absolute_path("checkpoints"),
                filename="{epoch}-{val_loss:.3f}",
                monitor="val_loss",
                save_top_k=3,
                mode="min",
                save_weights_only=True,
            ))
    if cfg.train.monitor:
        callbacks.append(DeviceStatsMonitor())
    logger = TensorBoardLogger(save_dir=to_absolute_path("log"),
                               name=cfg.name,
                               log_graph=True)
    trainer = Trainer(
        accelerator="auto",
        accumulate_grad_batches=cfg.train.acc,
        auto_lr_find=cfg.train.auto_lr,
        auto_scale_batch_size=cfg.train.auto_batch,
        callbacks=callbacks,
        detect_anomaly=True,
        devices="auto",
        fast_dev_run=cfg.train.fast_dev_run,
        logger=[logger],
        num_sanity_val_steps=2,
    )

    trainer.tune(model=model, datamodule=datamodule)
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)

    os.makedirs("onnx", exist_ok=True)
    model.to_onnx(file_path=to_absolute_path(os.path.join(
        "onnx", "model.onnx")),
                  export_params=True)


if __name__ == "__main__":
    main()
