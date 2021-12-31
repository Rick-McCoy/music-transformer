import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from pytorch_lightning.callbacks import DeviceStatsMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer

from data.datamodule import MusicDataModule
from model.model import MusicModel


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig = None) -> None:
    datamodule = MusicDataModule(cfg)
    devices = "auto" if cfg.train.gpus == -1 else cfg.train.gpus

    batch_size = cfg.train.batch_size
    if cfg.train.auto_batch:
        batch_model = MusicModel(d_model=cfg.model.d_model,
                                 data_len=cfg.model.data_len,
                                 dropout=cfg.model.dropout,
                                 ff=cfg.model.ff,
                                 lr=cfg.train.lr,
                                 nhead=cfg.model.nhead,
                                 num_layers=cfg.model.num_layers,
                                 num_token=cfg.model.num_token,
                                 segments=cfg.model.segments)
        batch_trainer = Trainer(
            accelerator="auto",
            accumulate_grad_batches=2,
            detect_anomaly=True,
            devices=devices,
            precision=16,
        )
        try:
            batch_size = batch_trainer.tuner.scale_batch_size(
                model=batch_model, datamodule=datamodule)
            datamodule.batch_size = batch_size
        except RuntimeError:
            return
        del batch_model, batch_trainer

    if cfg.train.effective_batch_size > 0:
        accumulate = cfg.train.effective_batch_size // batch_size
        accumulate = max(accumulate, 1)
    else:
        accumulate = cfg.train.acc

    if cfg.train.auto_lr:
        lr_model = MusicModel(d_model=cfg.model.d_model,
                              data_len=cfg.model.data_len,
                              dropout=cfg.model.dropout,
                              ff=cfg.model.ff,
                              lr=cfg.train.lr,
                              nhead=cfg.model.nhead,
                              num_layers=cfg.model.num_layers,
                              num_token=cfg.model.num_token,
                              segments=cfg.model.segments)
        lr_trainer = Trainer(
            accelerator="auto",
            accumulate_grad_batches=accumulate,
            detect_anomaly=True,
            devices=devices,
            precision=16,
        )
        lr_finder = lr_trainer.tuner.lr_find(model=lr_model,
                                             datamodule=datamodule,
                                             max_lr=0.01)
        learning_rate = lr_finder.suggestion()
        del lr_model, lr_trainer
    else:
        learning_rate = cfg.train.lr

    model = MusicModel(d_model=cfg.model.d_model,
                       data_len=cfg.model.data_len,
                       dropout=cfg.model.dropout,
                       ff=cfg.model.ff,
                       lr=learning_rate,
                       nhead=cfg.model.nhead,
                       num_layers=cfg.model.num_layers,
                       num_token=cfg.model.num_token,
                       segments=cfg.model.segments)
    callbacks = []
    if cfg.train.checkpoint:
        callbacks.append(
            ModelCheckpoint(
                dirpath=to_absolute_path("checkpoints"),
                filename="epoch={epoch}-val_loss={val/loss:.3f}",
                monitor="val/loss",
                save_top_k=1,
                mode="min",
                auto_insert_metric_name=False,
                save_weights_only=True,
            ))
    if cfg.train.monitor:
        callbacks.append(DeviceStatsMonitor())
    if cfg.train.early_stopping:
        callbacks.append(EarlyStopping(monitor="val/loss", mode="min"))
    logger = WandbLogger(project="music-model", save_dir=to_absolute_path("."))
    max_time = None if cfg.train.max_time == "" else cfg.train.max_time
    trainer = Trainer(
        accelerator="auto",
        accumulate_grad_batches=accumulate,
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

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
