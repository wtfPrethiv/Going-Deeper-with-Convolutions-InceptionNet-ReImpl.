from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass
class CIFAR10DataModule:
    root: str | Path = "data/dataset"
    batch_size: int = 64
    num_workers: int = 4
    mean: list[float] = field(default_factory=lambda: [0.4914, 0.4822, 0.4465])
    std: list[float] = field(default_factory=lambda: [0.2023, 0.1994, 0.2010])
    random_crop_size: int = 32
    random_crop_padding: int = 4
    color_jitter: float = 0.2
    pin_memory: bool = True
    persistent_workers: bool = True

    @property
    def train_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.RandomCrop(self.random_crop_size, padding=self.random_crop_padding),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=self.color_jitter,
                contrast=self.color_jitter,
                saturation=self.color_jitter,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    @property
    def val_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def get_train_loader(self) -> DataLoader:
        dataset = datasets.CIFAR10(
            root=str(self.root), train=True, download=True,
            transform=self.train_transform,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def get_val_loader(self) -> DataLoader:
        dataset = datasets.CIFAR10(
            root=str(self.root), train=False, download=True,
            transform=self.val_transform,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        return self.get_train_loader(), self.get_val_loader()

    @classmethod
    def from_config(cls, cfg) -> "CIFAR10DataModule":
        return cls(
            root=cfg.data.root,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
            mean=cfg.data.mean,
            std=cfg.data.std,
            random_crop_size=cfg.data.random_crop_size,
            random_crop_padding=cfg.data.random_crop_padding,
            color_jitter=cfg.data.color_jitter,
        )

    CLASSES = (
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    )

    @property
    def num_classes(self) -> int:
        return len(self.CLASSES)