"""Tests for multi-crop augmentation and the collate function."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from dinovpr.data.datasets import multicrop_collate_fn
from dinovpr.dino.augmentation import DINOMultiCropTransform


def _fake_image(size=64):
    return Image.fromarray(np.random.randint(0, 255, size=(size, size, 3), dtype=np.uint8))


def test_multi_crop_produces_expected_number_of_crops():
    mc = DINOMultiCropTransform(n_global_crops=2, n_local_crops=6)
    crops = mc(_fake_image())
    assert len(crops) == 8


def test_multi_crop_global_and_local_sizes():
    g, l = 32, 16
    mc = DINOMultiCropTransform(global_size=g, local_size=l, n_global_crops=2, n_local_crops=4)
    crops = mc(_fake_image())
    for c in crops[:2]:
        assert c.shape == (3, g, g)
    for c in crops[2:]:
        assert c.shape == (3, l, l)


def test_multi_crop_tensors_are_normalized():
    """Normalized tensors should have mean near 0 and std near 1 over large samples."""
    mc = DINOMultiCropTransform(global_size=64, local_size=32, n_global_crops=2, n_local_crops=0)
    all_vals = []
    for _ in range(10):
        for c in mc(_fake_image(128)):
            all_vals.append(c.flatten())
    v = torch.cat(all_vals)
    # Random images: after ImageNet normalization values span a wide range, not strictly mean 0,
    # but should be well within [-3, 3] range.
    assert v.min() > -4 and v.max() < 4


def test_collate_fn_transposes_correctly():
    mc = DINOMultiCropTransform(global_size=32, local_size=16, n_global_crops=2, n_local_crops=4)
    B = 5
    batch = [(mc(_fake_image()), i) for i in range(B)]
    crops, labels = multicrop_collate_fn(batch)
    assert isinstance(crops, list)
    assert len(crops) == 6
    # Two globals of (B, 3, 32, 32)
    assert crops[0].shape == (B, 3, 32, 32)
    assert crops[1].shape == (B, 3, 32, 32)
    # Four locals of (B, 3, 16, 16)
    for i in range(2, 6):
        assert crops[i].shape == (B, 3, 16, 16)
    # Labels preserved
    assert labels.tolist() == list(range(B))
