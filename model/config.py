from pathlib import Path

from omegaconf import DictConfig
from hydra.utils import to_absolute_path


class MusicConfig:
    def __init__(self, cfg: DictConfig) -> None:
        self.acc: int = cfg.train.acc
        self.augment: bool = cfg.data.augment
        self.auto_lr: bool = cfg.train.auto_lr
        self.auto_batch_size: bool = cfg.train.auto_batch_size
        self.batch_size: int = cfg.train.batch_size
        self.checkpoint_dir: Path = Path(
            to_absolute_path(*cfg.train.checkpoint_dir))
        self.d_model: int = cfg.model.d_model
        self.data_len: int = cfg.data.data_len
        self.data_dir: Path = Path(to_absolute_path(*cfg.data.data_dir))
        self.dropout: float = cfg.model.dropout
        self.early_stop: bool = cfg.train.early_stop
        self.effective_batch_size: int = cfg.train.effective_batch_size
        self.fast_dev_run: bool = cfg.train.fast_dev_run
        self.filename_list: Path = Path(
            to_absolute_path(*cfg.data.filename_list))
        self.ff: int = cfg.model.ff
        self.gpus: int = cfg.train.gpus
        self.limit_batches: float = cfg.train.limit_batches
        self.lr: float = cfg.train.lr
        self.max_time: int = cfg.train.max_time
        self.monitor: bool = cfg.train.monitor
        self.nhead: int = cfg.model.nhead
        self.note_shift: int = cfg.data.note_shift
        self.num_layer: int = cfg.model.num_layer
        self.num_special: int = cfg.data.num_special
        self.num_program: int = cfg.data.num_program
        self.num_note: int = cfg.data.num_note
        self.num_velocity: int = cfg.data.num_velocity
        self.num_time_num: int = cfg.data.num_time_num
        self.num_time_den: int = cfg.data.num_time_den
        self.num_workers: int = cfg.train.num_workers
        self.process_dir: Path = Path(to_absolute_path(*cfg.data.process_dir))
        self.time_scale: float = cfg.data.time_scale
        self.velocity_scale: float = cfg.data.velocity_scale
