from typing import Optional

from omegaconf.dictconfig import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from data.utils import load_mnist, get_mnist


class SimpleDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.train.batch_size
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def prepare_data(self) -> None:
        load_mnist(self.cfg.data.path)

    def setup(self, stage: Optional[str] = None) -> None:
        (self.train_dataset,
         self.val_dataset), self.test_dataset = get_mnist(self.cfg.data.path)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.cfg.train.num_workers,
                          pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.cfg.train.num_workers,
                          pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.cfg.train.num_workers,
                          pin_memory=True)
