"""
Evaluation-only transforms.

Kept deliberately simple: resize the shorter side, center-crop to a square,
convert to tensor, and normalize with ImageNet statistics (the public DINOv2
weights expect this normalisation, and our mini-trained model uses the same
convention for consistency).
"""

from __future__ import annotations

from typing import Tuple

from torchvision import transforms as T


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_eval_transform(
    image_size: int = 224,
    resize_size: int | None = None,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> T.Compose:
    """Standard eval pipeline: resize -> center crop -> tensor -> normalize.

    Parameters
    ----------
    image_size : int
        Output side length of the centre crop.
    resize_size : int or None
        Short-side resize target. If None, defaults to ``image_size * 256 / 224``
        (i.e. the standard ImageNet eval convention, 8/7 ratio).
    """
    if resize_size is None:
        resize_size = int(round(image_size * 256 / 224))
    return T.Compose(
        [
            T.Resize(resize_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
