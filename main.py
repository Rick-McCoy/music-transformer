"""The main file.

Run this file with hydra overrides for a single run of training.

    Example:
        python main.py train.acc=2 train.batch_size=32"""
import gc
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from pytorch_lightning.callbacks import DeviceStatsMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer
import torch

from data.datamodule import MusicDataModule
from data.utils import Modifier
from model.model import MusicModel


@hydra.main(config_path="config", config_name="main")
def main(cfg: DictConfig = None) -> None:
    """The main function.

    Accepts a DictConfig and trains the model accordingly.
    See `config/main.yaml` for modifiable hyperparameters.

    Args:
        cfg: DictConfig housing all hyperparameters.

    Examples:
        >>> from hydra.utils import initialize, compose
        >>> with initialize(config_path=\"config\"):
        >>>     main(cfg=compose(config_name=\"main\", overrides=...))"""
    # Get batch size
    batch_size = cfg.train.batch_size
    # Initialize Modifier
    modifier = Modifier(num_special=cfg.data.num_special,
                        num_program=cfg.data.num_program,
                        num_note=cfg.data.num_note,
                        num_velocity=cfg.data.num_velocity,
                        num_time_num=cfg.data.num_time_num,
                        note_shift=cfg.data.note_shift,
                        velocity_scale=cfg.data.velocity_scale,
                        time_scale=cfg.data.time_scale)
    # Initialize DataModule
    datamodule = MusicDataModule(
        batch_size=batch_size,
        data_dir=Path(to_absolute_path(Path(*cfg.data.data_dir))),
        text_dir=Path(to_absolute_path(Path(*cfg.data.text_dir))),
        process_dir=Path(to_absolute_path(Path(*cfg.data.process_dir))),
        num_workers=cfg.train.num_workers,
        data_len=cfg.model.data_len,
        augment=cfg.data.augment,
        modifier=modifier)
    # Get GPU
    devices = "auto" if cfg.train.gpus == -1 else cfg.train.gpus
    # Calculate num_token
    num_token = cfg.data.num_special + cfg.data.num_program + cfg.data.num_note \
        + cfg.data.num_velocity + cfg.data.num_time_num + cfg.data.num_time_denum

    # If auto_batch, search for maximum possible batch size
    if cfg.train.auto_batch:
        # Initialize temporary model
        batch_model = MusicModel(d_model=cfg.model.d_model,
                                 data_len=cfg.model.data_len,
                                 dropout=cfg.model.dropout,
                                 ff=cfg.model.ff,
                                 lr=cfg.train.lr,
                                 nhead=cfg.model.nhead,
                                 num_layer=cfg.model.num_layer,
                                 num_temp=cfg.model.num_temp,
                                 num_token=num_token,
                                 segments=cfg.model.segments)
        # Initialize temporary trainer
        batch_trainer = Trainer(accelerator="auto",
                                accumulate_grad_batches=2,
                                detect_anomaly=True,
                                devices=devices,
                                precision=16)
        # Try to scale batch size
        # Prone to CUDA OOM
        try:
            batch_size = batch_trainer.tuner.scale_batch_size(
                model=batch_model, datamodule=datamodule)
            datamodule.batch_size = batch_size
        except RuntimeError:
            # If batch size scaling fails, model is likely too large
            # Or pytorch lightning is bugged
            return
        # Clean up as best as possible
        del batch_model, batch_trainer
        torch.cuda.empty_cache()
        gc.collect()

    # Set accumulate according to effective batch size
    if cfg.train.effective_batch_size > 0:
        accumulate = cfg.train.effective_batch_size // batch_size
        accumulate = max(accumulate, 1)
    else:
        accumulate = cfg.train.acc

    # If auto_lr, search for optimal LR
    if cfg.train.auto_lr:
        # Initialize temporary model
        lr_model = MusicModel(d_model=cfg.model.d_model,
                              data_len=cfg.model.data_len,
                              dropout=cfg.model.dropout,
                              ff=cfg.model.ff,
                              lr=cfg.train.lr,
                              nhead=cfg.model.nhead,
                              num_layer=cfg.model.num_layer,
                              num_temp=cfg.model.num_temp,
                              num_token=num_token,
                              segments=cfg.model.segments)
        # Initialize temporary trainer
        lr_trainer = Trainer(accelerator="auto",
                             accumulate_grad_batches=accumulate,
                             detect_anomaly=True,
                             devices=devices,
                             precision=16)
        # Get lr tuner
        lr_finder = lr_trainer.tuner.lr_find(model=lr_model,
                                             datamodule=datamodule,
                                             max_lr=0.01)
        # Calculate optimal lr
        learning_rate = lr_finder.suggestion()
        print(f"Learning rate: {learning_rate}")
        # Clean up as best as possible
        del lr_model, lr_trainer
        torch.cuda.empty_cache()
        gc.collect()
    else:
        learning_rate = cfg.train.lr

    # Initialize actual model
    model = MusicModel(d_model=cfg.model.d_model,
                       data_len=cfg.model.data_len,
                       dropout=cfg.model.dropout,
                       ff=cfg.model.ff,
                       lr=cfg.train.lr,
                       nhead=cfg.model.nhead,
                       num_layer=cfg.model.num_layer,
                       num_temp=cfg.model.num_temp,
                       num_token=num_token,
                       segments=cfg.model.segments)
    # Set callbacks
    callbacks = []
    if cfg.train.checkpoint:
        # Set checkpoint callback
        # Saves checkpoint with smallest validation loss
        callbacks.append(
            ModelCheckpoint(dirpath=to_absolute_path("checkpoints"),
                            filename="epoch={epoch}-val_loss={val/loss:.3f}",
                            monitor="val/loss",
                            save_top_k=1,
                            mode="min",
                            auto_insert_metric_name=False,
                            save_weights_only=True))
    if cfg.train.monitor:
        # Set device stat monitor
        callbacks.append(DeviceStatsMonitor())
    if cfg.train.early_stopping:
        # Set early stopping callback
        # Moonitors validation loss, stops training when improvements stop
        callbacks.append(EarlyStopping(monitor="val/loss", mode="min"))
    # Setup W&B Logger
    logger = WandbLogger(project="music-transformer",
                         save_dir=to_absolute_path("."))
    # Stops after max_time elapses
    max_time = None if cfg.train.max_time == "" else cfg.train.max_time
    # Initialize actual trainer
    trainer = Trainer(accelerator="auto",
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
                      precision=16)

    # Run optimization & testing
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
