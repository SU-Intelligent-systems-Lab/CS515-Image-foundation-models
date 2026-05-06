"""
DINO multi-crop augmentation.

For each image, the pipeline produces:
  * ``n_global_crops`` large crops (default 2) of size ``global_size``,
    covering a substantial portion of the image (default 40%–100% area),
  * ``n_local_crops`` small crops (default 6) of size ``local_size``,
    covering a smaller portion (default 5%–40% area).

The teacher processes only global crops; the student processes all crops.
The asymmetry is what encourages ``local-to-global`` correspondence learning.

We keep augmentations lightweight for our CIFAR-scale mini-reimplementation
(no ImageNet-scale solarization / Gaussian blur stacks), matching the scale
at which we train.

Reference
---------
Caron et al. 2021, Section 3 and Appendix. The original values in the paper
are 224px global / 96px local, with global scale (0.4, 1.0) and local scale
(0.05, 0.4). Our defaults are scaled down to 32px global / 16px local for
CIFAR-100.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from PIL import Image
from torchvision import transforms as T


# Standard ImageNet normalization. (We use it even on CIFAR because the public
# DINOv2 weights we compare against expect it.)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class GaussianBlur:
    """Random Gaussian blur with probability p.

    Reimplemented here (rather than relying on any particular torchvision
    version) because the transform contract changes across torchvision
    releases. Operates on PIL images.
    """

    def __init__(self, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img: Image.Image) -> Image.Image:
        if torch.rand(1).item() > self.p:
            return img
        from PIL import ImageFilter
        radius = self.radius_min + torch.rand(1).item() * (self.radius_max - self.radius_min)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


class Solarize:
    """Random solarization with probability p. Operates on PIL images."""

    def __init__(self, p: float = 0.2, threshold: int = 128):
        self.p = p
        self.threshold = threshold

    def __call__(self, img: Image.Image) -> Image.Image:
        if torch.rand(1).item() > self.p:
            return img
        from PIL import ImageOps
        return ImageOps.solarize(img, threshold=self.threshold)


class DINOMultiCropTransform:
    """Multi-crop transform pipeline for DINO-style training.

    On call, returns a list of ``n_global_crops + n_local_crops`` tensors,
    all with 3 channels. The list ordering is:

        [global_0, global_1, ..., local_0, local_1, ...]

    Parameters
    ----------
    global_size : int
        Output side length (pixels) of global crops.
    local_size : int
        Output side length of local crops.
    n_global_crops : int
        Number of global crops (default 2, per DINO).
    n_local_crops : int
        Number of local crops (default 6).
    global_scale : (float, float)
        ``RandomResizedCrop`` area-fraction range for global crops.
    local_scale : (float, float)
        Same for local crops.
    color_jitter : (float, float, float, float)
        Brightness, contrast, saturation, hue amplitudes.
    mean, std : tuples
        Normalization stats.
    """

    def __init__(
        self,
        global_size: int = 32,
        local_size: int = 16,
        n_global_crops: int = 2,
        n_local_crops: int = 6,
        global_scale: Tuple[float, float] = (0.4, 1.0),
        local_scale: Tuple[float, float] = (0.05, 0.4),
        color_jitter: Tuple[float, float, float, float] = (0.4, 0.4, 0.2, 0.1),
        mean: Tuple[float, float, float] = IMAGENET_MEAN,
        std: Tuple[float, float, float] = IMAGENET_STD,
    ):
        self.n_global_crops = n_global_crops
        self.n_local_crops = n_local_crops

        flip_and_color = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(*color_jitter)], p=0.8),
                T.RandomGrayscale(p=0.2),
            ]
        )
        normalize = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])

        # First global crop: blur always on.
        self.global_transform_1 = T.Compose(
            [
                T.RandomResizedCrop(
                    global_size, scale=global_scale, interpolation=T.InterpolationMode.BICUBIC
                ),
                flip_and_color,
                GaussianBlur(p=1.0),
                normalize,
            ]
        )
        # Second global crop: blur with p=0.1 and solarize with p=0.2.
        self.global_transform_2 = T.Compose(
            [
                T.RandomResizedCrop(
                    global_size, scale=global_scale, interpolation=T.InterpolationMode.BICUBIC
                ),
                flip_and_color,
                GaussianBlur(p=0.1),
                Solarize(p=0.2),
                normalize,
            ]
        )
        # Local crops: small, blur with p=0.5.
        self.local_transform = T.Compose(
            [
                T.RandomResizedCrop(
                    local_size, scale=local_scale, interpolation=T.InterpolationMode.BICUBIC
                ),
                flip_and_color,
                GaussianBlur(p=0.5),
                normalize,
            ]
        )

    def __call__(self, img: Image.Image) -> List[torch.Tensor]:
        crops: List[torch.Tensor] = []
        # Global crops
        crops.append(self.global_transform_1(img))
        for _ in range(self.n_global_crops - 1):
            crops.append(self.global_transform_2(img))
        # Local crops
        for _ in range(self.n_local_crops):
            crops.append(self.local_transform(img))
        return crops
