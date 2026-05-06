"""
From-scratch reimplementation of the DINO self-distillation training loop.

This subpackage is intentionally minimal and pedagogical: it reimplements the
core mechanism of DINO (Caron et al., 2021) so that the project's report can
demonstrate end-to-end understanding of the training procedure. It is **not**
intended to reach the quality of the publicly released DINOv2/v3 checkpoints,
which were trained on orders-of-magnitude more data.

Submodules
----------
model            -- Vision Transformer + DINO projection head
loss             -- DINO cross-entropy loss with centering and sharpening
augmentation     -- Multi-crop data augmentation pipeline
teacher_student  -- EMA teacher update + momentum schedules
train            -- One-step / one-epoch training utilities
"""

from .model import VisionTransformer, DINOHead, DINOModel  # noqa: F401
from .loss import DINOLoss  # noqa: F401
from .augmentation import DINOMultiCropTransform  # noqa: F401
from .teacher_student import (  # noqa: F401
    update_teacher_ema,
    cosine_schedule,
    momentum_schedule,
)
