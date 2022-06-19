"""
Consolidates all hydra configs into one class.
"""

from pathlib import Path

from omegaconf import DictConfig


class CustomConfig:
    """
    Flattens all hydra configuration files.
    See configuration files under config/ for more information regarding individual parameters.
    """

    def __init__(self, cfg: DictConfig = None):
        if cfg is None:
            raise ValueError("Initialize with a hydra config.")
        self.cfg = cfg

        self.checkpoint_dir = Path(self.cfg.data.checkpoint_dir)
        self.data_dir = Path(self.cfg.data.data_dir)
        self.file_dir = Path(self.cfg.data.file_dir)
        self.log_dir = Path(self.cfg.data.log_dir)
        self.process_dir = Path(self.cfg.data.process_dir)

        self.d_model: int = self.cfg.model.d_model
        self.data_len: int = self.cfg.model.data_len
        self.dropout: float = self.cfg.model.dropout
        self.feed_forward: bool = self.cfg.model.ff
        self.nhead: int = self.cfg.model.nhead
        self.num_layers: int = self.cfg.model.num_layers
        self.num_special: int = self.cfg.model.num_special
        self.num_program: int = self.cfg.model.num_program
        self.num_note: int = self.cfg.model.num_note
        self.num_tick: int = self.cfg.model.num_tick
        self.num_tokens: int = self.num_special + self.num_program + self.num_note + self.num_tick
        self.segments: int = self.cfg.model.segments

        self.acc: int = self.cfg.train.acc
        self.auto_batch: bool = self.cfg.train.auto_batch
        self.auto_lr: bool = self.cfg.train.auto_lr
        self.batch_size: int = self.cfg.train.batch_size
        self.checkpoint: bool = self.cfg.train.checkpoint
        self.early_stop: bool = self.cfg.train.early_stop
        self.effective_batch_size: int = self.cfg.train.effective_batch_size
        self.fast_dev_run: bool = self.cfg.train.fast_dev_run
        self.gpus: int = self.cfg.train.gpus
        self.limit_batches: int = self.cfg.train.limit_batches
        self.learning_rate: float = self.cfg.train.lr
        self.max_epochs: int = self.cfg.train.max_epochs
        self.max_time: str = self.cfg.train.max_time
        self.monitor: bool = self.cfg.train.monitor
        self.num_workers: int = self.cfg.train.num_workers