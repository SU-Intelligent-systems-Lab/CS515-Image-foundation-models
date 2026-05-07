"""
data_utils.py — Dataset Loading for Semantic Segmentation
=========================================================

This module provides data loading utilities for the two primary segmentation
benchmarks used in the DINOv3 paper:

1. **ADE20k** (Zhou et al., 2017)
   - 150 semantic categories
   - 20,210 training images, 2,000 validation images
   - Categories span indoor/outdoor scenes (bed, tree, sky, wall, etc.)
   - Download: http://sceneparsing.csail.mit.edu/

2. **Pascal VOC 2012** (Everingham et al., 2012)
   - 21 semantic categories (20 object classes + background)
   - 1,464 training images, 1,449 validation images
   - Categories: person, car, dog, cat, aeroplane, etc.
   - Download: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

Data Preprocessing (from DINOv3 Appendix D.1)
---------------------------------------------
- Training: Random resize (0.5× to 2.0×), random crop to 512×512,
  horizontal flip with probability 0.5
- Evaluation: Resize shorter side to ``img_size``, sliding window inference
- Normalization: ImageNet mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
"""

import logging
import os
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

logger = logging.getLogger(__name__)


# ImageNet normalization constants (used by DINOv3)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ─────────────────────────────────────────────────────────────────────────────
# ADE20k Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ADE20kDataset(Dataset):
    """
    ADE20k Semantic Segmentation Dataset.

    Expected directory structure:
    ::

        ADEChallengeData2016/
        ├── images/
        │   ├── training/         # 20,210 images
        │   │   ├── ADE_train_00000001.jpg
        │   │   └── ...
        │   └── validation/       # 2,000 images
        │       ├── ADE_val_00000001.jpg
        │       └── ...
        └── annotations/
            ├── training/         # Segmentation masks (PNG)
            │   ├── ADE_train_00000001.png
            │   └── ...
            └── validation/
                ├── ADE_val_00000001.png
                └── ...

    Label format:
    - Pixel values in [0, 150] where 0 = background/unlabeled
    - During evaluation with ``reduce_zero_label=True``, label 0 is mapped to
      ignore_index (255), and labels 1-150 are shifted to 0-149.
    - This results in 150 evaluated classes.

    Parameters
    ----------
    root : str
        Path to ADEChallengeData2016 directory.
    split : str
        "train" or "val".
    transform : callable, optional
        Joint transform for image and mask.
    """

    def __init__(self, root: str, split: str = "train", transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        split_dir = "training" if split == "train" else "validation"
        self.img_dir = os.path.join(root, "images", split_dir)
        self.ann_dir = os.path.join(root, "annotations", split_dir)

        self.images = sorted(os.listdir(self.img_dir))
        logger.info(f"ADE20k {split}: {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Returns
        -------
        image : Tensor of shape (3, H, W) — normalized RGB image
        target : Tensor of shape (H, W) — class indices per pixel
        """
        img_name = self.images[idx]
        ann_name = img_name.replace(".jpg", ".png")

        image = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
        target = Image.open(os.path.join(self.ann_dir, ann_name))

        if self.transform:
            image, target = self.transform(image, target)

        return image, target


# ─────────────────────────────────────────────────────────────────────────────
# Pascal VOC Dataset
# ─────────────────────────────────────────────────────────────────────────────

class VOCSegmentationDataset(Dataset):
    """
    Pascal VOC 2012 Semantic Segmentation Dataset.

    Expected directory structure:
    ::

        VOCdevkit/VOC2012/
        ├── JPEGImages/            # RGB images
        │   ├── 2007_000027.jpg
        │   └── ...
        ├── SegmentationClass/     # Segmentation masks (PNG)
        │   ├── 2007_000032.png
        │   └── ...
        └── ImageSets/Segmentation/
            ├── train.txt          # Image IDs for training
            ├── val.txt            # Image IDs for validation
            └── trainval.txt

    Label format:
    - 21 classes (0=background, 1-20=object classes)
    - Pixel value 255 = boundary/ignore

    Parameters
    ----------
    root : str
        Path to VOCdevkit/VOC2012 directory.
    split : str
        "train" or "val".
    transform : callable, optional
        Joint transform for image and mask.
    """

    def __init__(self, root: str, split: str = "train", transform=None):
        self.root = root
        self.transform = transform

        split_file = os.path.join(root, "ImageSets", "Segmentation", f"{split}.txt")
        with open(split_file) as f:
            self.image_ids = [line.strip() for line in f.readlines()]

        self.img_dir = os.path.join(root, "JPEGImages")
        self.ann_dir = os.path.join(root, "SegmentationClass")

        logger.info(f"Pascal VOC {split}: {len(self.image_ids)} images")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]

        image = Image.open(os.path.join(self.img_dir, f"{img_id}.jpg")).convert("RGB")
        target = Image.open(os.path.join(self.ann_dir, f"{img_id}.png"))

        if self.transform:
            image, target = self.transform(image, target)

        return image, target


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────

class SegmentationTrainTransform:
    """
    Training transform for semantic segmentation.

    Applies random resizing, random cropping, horizontal flipping, and
    ImageNet normalization — matching the DINOv3 training protocol.

    Parameters
    ----------
    img_size : int
        Target crop size (both height and width).
    scale_range : tuple
        (min_scale, max_scale) for random resizing.

    Applied Operations
    ------------------
    1. Random resize: scale the shorter side by a random factor in scale_range
    2. Random crop: extract a (img_size × img_size) patch
    3. Random horizontal flip with 50% probability
    4. Convert to tensor and normalize with ImageNet stats
    """

    def __init__(self, img_size: int = 512, scale_range: Tuple[float, float] = (0.5, 2.0)):
        self.img_size = img_size
        self.scale_range = scale_range

    def __call__(self, image: Image.Image, target: Image.Image):
        # Random scale
        scale = np.random.uniform(*self.scale_range)
        w, h = image.size
        new_h, new_w = int(h * scale), int(w * scale)
        image = TF.resize(image, (new_h, new_w), interpolation=TF.InterpolationMode.BILINEAR)
        target = TF.resize(target, (new_h, new_w), interpolation=TF.InterpolationMode.NEAREST)

        # Random crop
        crop_h, crop_w = self.img_size, self.img_size
        if new_h < crop_h or new_w < crop_w:
            # Pad if image is smaller than crop
            pad_h = max(crop_h - new_h, 0)
            pad_w = max(crop_w - new_w, 0)
            image = TF.pad(image, (0, 0, pad_w, pad_h), fill=0)
            target = TF.pad(target, (0, 0, pad_w, pad_h), fill=255)
            new_h, new_w = max(new_h, crop_h), max(new_w, crop_w)

        top = np.random.randint(0, new_h - crop_h + 1)
        left = np.random.randint(0, new_w - crop_w + 1)
        image = TF.crop(image, top, left, crop_h, crop_w)
        target = TF.crop(target, top, left, crop_h, crop_w)

        # Random horizontal flip
        if np.random.random() > 0.5:
            image = TF.hflip(image)
            target = TF.hflip(target)

        # To tensor and normalize
        image = TF.to_tensor(image)  # (3, H, W), values in [0, 1]
        image = TF.normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        target = torch.from_numpy(np.array(target)).long()

        return image, target


class SegmentationEvalTransform:
    """
    Evaluation transform for semantic segmentation.

    Simply resizes the image to a fixed size and normalizes it.
    The ground truth mask is resized with nearest-neighbor interpolation
    to preserve class labels.

    Parameters
    ----------
    img_size : int
        Target size for the shorter side.
    """

    def __init__(self, img_size: int = 512):
        self.img_size = img_size

    def __call__(self, image: Image.Image, target: Image.Image):
        image = TF.resize(image, (self.img_size, self.img_size), interpolation=TF.InterpolationMode.BILINEAR)
        target = TF.resize(target, (self.img_size, self.img_size), interpolation=TF.InterpolationMode.NEAREST)

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        target = torch.from_numpy(np.array(target)).long()

        return image, target


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_segmentation_dataloaders(
    dataset_name: str = "ade20k",
    dataset_root: str = "data/ADEChallengeData2016",
    img_size: int = 512,
    batch_size: int = 2,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders for segmentation.

    Parameters
    ----------
    dataset_name : str
        "ade20k" or "voc".
    dataset_root : str
        Path to the dataset root directory.
    img_size : int
        Image size for training (crop size) and evaluation.
    batch_size : int
        Batch size for training. Validation always uses batch_size=1
        (required for sliding window inference).
    num_workers : int
        Number of data loading workers.

    Returns
    -------
    tuple of (DataLoader, DataLoader)
        (train_loader, val_loader)

    Example
    -------
    >>> train_loader, val_loader = get_segmentation_dataloaders(
    ...     dataset_name="ade20k",
    ...     dataset_root="data/ADEChallengeData2016",
    ...     img_size=512,
    ... )
    >>> images, targets = next(iter(train_loader))
    >>> print(images.shape, targets.shape)
    torch.Size([2, 3, 512, 512]) torch.Size([2, 512, 512])
    """
    train_transform = SegmentationTrainTransform(img_size=img_size)
    val_transform = SegmentationEvalTransform(img_size=img_size)

    if dataset_name == "ade20k":
        train_dataset = ADE20kDataset(dataset_root, split="train", transform=train_transform)
        val_dataset = ADE20kDataset(dataset_root, split="val", transform=val_transform)
    elif dataset_name == "voc":
        train_dataset = VOCSegmentationDataset(dataset_root, split="train", transform=train_transform)
        val_dataset = VOCSegmentationDataset(dataset_root, split="val", transform=val_transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'ade20k' or 'voc'.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Must be 1 for sliding window inference
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
