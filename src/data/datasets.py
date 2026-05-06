"""
Dataset wrappers for the project.

Two modes are supported for every dataset:

1. **Multi-crop mode** (for DINO-style SSL training):
   ``__getitem__`` returns ``(List[Tensor], int)`` where the list contains all
   crops produced by a :class:`DINOMultiCropTransform`, and the label is kept
   only so that evaluation callbacks can use it (the SSL loss never touches
   it).

2. **Eval mode** (for kNN / linear probing / retrieval):
   ``__getitem__`` returns ``(Tensor, int)`` via a standard transform.

Builders
--------
build_cifar100
    CIFAR-100 via torchvision. Small enough to fit in memory on disk. Ideal
    for the mini DINO training loop. Images are 32x32 RGB.

build_imagenet100
    Expects the user to have pre-downloaded the standard "ImageNet-100" subset
    (100 classes from ImageNet-1k) into a folder with ``train/`` and ``val/``
    subfolders in ImageFolder format. Used for higher-quality probing.

build_eval_loader
    Convenience for constructing an evaluation DataLoader with the standard
    eval transform.
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets as tvd

from .transforms import build_eval_transform


# -----------------------------------------------------------------------------
# Generic multi-crop dataset wrapper
# -----------------------------------------------------------------------------

class MultiCropDataset(Dataset):
    """Wrap any PIL-returning dataset to apply a multi-crop transform.

    The wrapped dataset's ``__getitem__`` must return ``(PIL.Image, label)``.
    """

    def __init__(self, base: Dataset, multi_crop_transform: Callable):
        self.base = base
        self.transform = multi_crop_transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], int]:
        img, label = self.base[idx]
        if not isinstance(img, Image.Image):
            # torchvision's CIFAR returns numpy arrays for some versions
            img = Image.fromarray(img) if not isinstance(img, Image.Image) else img
        crops = self.transform(img)
        return crops, int(label)


# -----------------------------------------------------------------------------
# Dataset builders
# -----------------------------------------------------------------------------

def _cifar100_pil_dataset(root: str, train: bool, download: bool) -> tvd.CIFAR100:
    """CIFAR-100 without any transform - yields PIL images for later transform."""
    # `transform=None` yields PIL images (not tensors), which our pipeline wants.
    return tvd.CIFAR100(root=root, train=train, download=download, transform=None)


def build_cifar100(
    root: str,
    train: bool = True,
    multi_crop_transform: Optional[Callable] = None,
    eval_transform: Optional[Callable] = None,
    download: bool = True,
) -> Dataset:
    """Build a CIFAR-100 dataset in either multi-crop SSL mode or eval mode.

    Parameters
    ----------
    root : str
        Directory where CIFAR-100 will be stored / is stored.
    train : bool
        True for the train split, False for the test split.
    multi_crop_transform : callable or None
        If given, SSL mode: each item is ``(list[Tensor], label)``.
    eval_transform : callable or None
        If given (and ``multi_crop_transform`` is None), eval mode:
        each item is ``(Tensor, label)``. If both are None, the default
        eval transform is used.
    """
    base = _cifar100_pil_dataset(root, train=train, download=download)

    if multi_crop_transform is not None:
        if eval_transform is not None:
            raise ValueError("Pass either multi_crop_transform OR eval_transform, not both.")
        return MultiCropDataset(base, multi_crop_transform)

    # Eval mode: attach the transform directly to the torchvision dataset.
    eval_transform = eval_transform or build_eval_transform(image_size=32, resize_size=32)
    base.transform = eval_transform
    return base


def build_imagenet100(
    root: str,
    train: bool = True,
    multi_crop_transform: Optional[Callable] = None,
    eval_transform: Optional[Callable] = None,
) -> Dataset:
    """Build ImageNet-100 (ImageFolder-format) in multi-crop or eval mode.

    ``root`` should point to a directory containing ``train/`` and ``val/``
    subfolders, each with one subdirectory per class.
    """
    split = "train" if train else "val"
    split_dir = os.path.join(root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(
            f"Expected ImageFolder-format directory at {split_dir!r}. "
            f"For ImageNet-100, see e.g. https://www.kaggle.com/datasets/ambityga/imagenet100"
        )

    if multi_crop_transform is not None:
        if eval_transform is not None:
            raise ValueError("Pass either multi_crop_transform OR eval_transform, not both.")
        base = tvd.ImageFolder(split_dir, transform=None)
        return MultiCropDataset(base, multi_crop_transform)

    eval_transform = eval_transform or build_eval_transform(image_size=224)
    return tvd.ImageFolder(split_dir, transform=eval_transform)


def build_eval_loader(
    dataset: Dataset,
    batch_size: int = 128,
    num_workers: int = 4,
    shuffle: bool = False,
    pin_memory: bool = True,
) -> DataLoader:
    """Wrap a dataset in a DataLoader with sensible defaults for evaluation."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


# -----------------------------------------------------------------------------
# Custom collate for SSL mode
# -----------------------------------------------------------------------------

def multicrop_collate_fn(
    batch: List[Tuple[List[torch.Tensor], int]],
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Custom collate: stack the same-indexed crop across the batch into a single tensor.

    The dataset yields ``(list_of_crops, label)`` for each sample; this collate
    transposes so that the dataloader yields ``(list_of_batched_crops, labels)``
    where ``list_of_batched_crops[i]`` has shape ``(B, 3, H_i, W_i)``.
    """
    n_crops = len(batch[0][0])
    crops_stacked = [torch.stack([sample[0][i] for sample in batch], dim=0) for i in range(n_crops)]
    labels = torch.tensor([sample[1] for sample in batch], dtype=torch.long)
    return crops_stacked, labels
