"""
Dataset loaders and transforms for the project.

Submodules
----------
datasets
    Wrappers around CIFAR-100 and ImageNet-100 that yield DINO multi-crop
    tuples when a multi-crop transform is provided, and standard single-view
    tensors otherwise (used for evaluation / linear probing).
transforms
    Evaluation-only transforms (standard resize + center crop + normalize).
"""

from .datasets import (  # noqa: F401
    MultiCropDataset,
    build_cifar100,
    build_imagenet100,
    build_eval_loader,
)
from .transforms import build_eval_transform  # noqa: F401
