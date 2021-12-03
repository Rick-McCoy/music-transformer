from typing import Tuple

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split

from torchvision import transforms
from torchvision.datasets import MNIST

from hydra.utils import to_absolute_path


def load_mnist(path: str) -> None:
    MNIST(root=to_absolute_path(path), train=True, download=True)
    MNIST(root=to_absolute_path(path), train=False, download=False)


def get_mnist(path: str) -> Tuple[Dataset, Dataset, Dataset]:
    mnist_transforms = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),
    ])
    train_val_dataset = MNIST(root=to_absolute_path(path),
                              train=True,
                              transform=mnist_transforms)
    test_dataset = MNIST(root=to_absolute_path(path),
                         train=False,
                         transform=mnist_transforms)
    train_len = int(len(train_val_dataset) * 0.9)
    val_len = len(train_val_dataset) - train_len

    return random_split(train_val_dataset, [train_len, val_len]), test_dataset
