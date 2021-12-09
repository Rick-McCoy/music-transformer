import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import DeviceStatsMonitor, ModelCheckpoint

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
        accumulate_grad_batches=cfg.train.acc,
        auto_lr_find=cfg.train.auto_lr,
        auto_scale_batch_size=cfg.train.auto_batch,
        auto_select_gpus=True,
        callbacks=callbacks,
        detect_anomaly=True,
        enable_checkpointing=True,
        fast_dev_run=cfg.train.fast_dev_run,
        gpus=cfg.train.gpus,
        logger=[logger],
        num_sanity_val_steps=2,
    )

    trainer.tune(model=model, datamodule=datamodule)
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
