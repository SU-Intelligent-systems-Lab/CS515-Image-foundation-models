"""
linear_probe.py — Training a Linear Segmentation Head on Frozen DINOv3 Features
================================================================================

This module implements **Section 6.1.2** of the DINOv3 paper: Dense Linear Probing
for semantic segmentation.

Concept
-------
The idea is simple but powerful: freeze the entire DINOv3 backbone and train ONLY
a single linear layer (1×1 convolution) on top of the patch features to predict
per-pixel class labels. If a linear classifier can achieve strong segmentation
performance, this proves the backbone's dense features are already well-structured
and semantically meaningful — without any task-specific fine-tuning.

Architecture Diagram
--------------------
::

    Input Image (B, 3, 512, 512)
           │
           ▼
    ┌──────────────────────┐
    │  DINOv3 Backbone     │  ← FROZEN (no gradients)
    │  (e.g. ViT-L/16)    │
    │  Extracts patch      │
    │  features from the   │
    │  last transformer    │
    │  block               │
    └──────────────────────┘
           │
           ▼
    Patch Features (B, 1024, 32, 32)
           │   ← 32×32 spatial grid, 1024-dim per patch
           │
           ▼
    ┌──────────────────────┐
    │  BatchNorm (1024)    │  ← Normalizes feature distribution
    └──────────────────────┘
           │
           ▼
    ┌──────────────────────┐
    │  Conv2d(1024 → C)    │  ← TRAINABLE: 1×1 conv, C = num_classes
    │  (the linear probe)  │     For ADE20k: C = 150
    └──────────────────────┘
           │
           ▼
    ┌──────────────────────┐
    │  Bilinear Upsample   │  ← Upscale from 32×32 to original resolution
    └──────────────────────┘
           │
           ▼
    Prediction (B, C, H, W)  ← Per-pixel class logits

Training Details (from the paper, Appendix D.1)
-----------------------------------------------
- Optimizer: AdamW
- Learning rate: sweep over {1e-4, 3e-4, 1e-3}
- Weight decay: sweep over {1e-4, 1e-3}
- Scheduler: Warmup + OneCycleLR (cosine annealing)
- Batch size: 2 per GPU
- Total iterations: 40,000 (ADE20k)
- Input resolution: 512×512 for patch size 16
- Evaluation: Sliding window (crop=512, stride=341)
- Backbone features: After layer normalization, with learned batch normalization

Key Insight
-----------
The linear probe achieves 54.9 mIoU on ADE20k with ViT-L — remarkably close to
the 63.0 mIoU achieved by the full Mask2Former decoder (which has 927M trainable
parameters vs ~150K for the linear probe). This demonstrates that DINOv3's frozen
features are already highly structured for dense prediction.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Linear Segmentation Head
# ─────────────────────────────────────────────────────────────────────────────

class LinearSegmentationHead(nn.Module):
    """
    A lightweight linear segmentation head for probing frozen backbone features.

    This is a direct re-implementation of DINOv3's ``LinearHead`` from
    ``dinov3/eval/segmentation/models/heads/linear_head.py``.

    The head consists of:
    1. Optional BatchNorm to normalize the feature distribution
    2. A 1×1 convolution that maps feature dimensions to class logits
    3. Dropout for regularization during training

    Parameters
    ----------
    in_channels : list of int
        Feature dimensions from each extracted backbone layer.
        For single-layer extraction (last layer of ViT-L): [1024]
        For multi-layer extraction (4 layers of ViT-L): [1024, 1024, 1024, 1024]
    num_classes : int
        Number of segmentation classes. ADE20k: 150, VOC: 21.
    use_batchnorm : bool
        Whether to apply SyncBatchNorm before the linear layer.
        This is important for stabilizing features from the frozen backbone.
    dropout : float
        Dropout probability applied during training only.

    Input
    -----
    features : list of torch.Tensor
        List of feature maps from the backbone.
        Each tensor has shape (B, C, H, W) where:
        - B = batch size
        - C = embed_dim (e.g., 1024 for ViT-L)
        - H, W = spatial dimensions (e.g., 32×32 for 512px input with patch16)

    Output
    ------
    torch.Tensor
        Class logits with shape (B, num_classes, H, W)
        where H, W match the input feature map resolution (NOT the original image).
        Upsampling to the original resolution is done externally.

    Example
    -------
    >>> head = LinearSegmentationHead(in_channels=[1024], num_classes=150)
    >>> features = [torch.randn(2, 1024, 32, 32)]  # ViT-L features for 512×512 input
    >>> logits = head(features)
    >>> print(logits.shape)  # (2, 150, 32, 32)
    """

    def __init__(
        self,
        in_channels: list,
        num_classes: int = 150,
        use_batchnorm: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.total_channels = sum(in_channels)
        self.num_classes = num_classes

        # BatchNorm normalizes the frozen backbone features.
        # SyncBatchNorm is used for multi-GPU training to compute statistics
        # across all GPUs. Falls back to regular BatchNorm for single GPU.
        self.batchnorm = (
            nn.SyncBatchNorm(self.total_channels)
            if use_batchnorm
            else nn.Identity()
        )

        # The actual "linear probe": a 1×1 convolution
        # This is mathematically equivalent to a linear layer applied independently
        # to each spatial location, mapping (C,) → (num_classes,)
        self.classifier = nn.Conv2d(
            self.total_channels, num_classes, kernel_size=1, padding=0, stride=1
        )

        # Dropout for regularization during training
        self.dropout = nn.Dropout2d(dropout)

        # Weight initialization
        nn.init.normal_(self.classifier.weight, mean=0, std=0.01)
        nn.init.constant_(self.classifier.bias, 0)

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"LinearSegmentationHead: {n_params:,} parameters "
            f"({self.total_channels} → {num_classes} classes)"
        )

    def forward(self, features: list) -> torch.Tensor:
        """
        Forward pass during training (includes dropout).

        Parameters
        ----------
        features : list of Tensor
            Backbone features. Each element has shape (B, C_i, H, W).
            If multiple layers are provided, they are concatenated along
            the channel dimension after being resized to the same spatial
            resolution (matching the first element).

        Returns
        -------
        Tensor of shape (B, num_classes, H, W)
        """
        # If multiple feature maps are provided (from different layers),
        # interpolate them all to the same spatial size and concatenate
        if len(features) > 1:
            target_size = features[0].shape[2:]  # (H, W) of first feature map
            features = [
                F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
                for f in features
            ]
        x = torch.cat(features, dim=1)  # (B, total_channels, H, W)

        x = self.dropout(x)
        x = self.batchnorm(x)
        x = self.classifier(x)  # (B, num_classes, H, W)
        return x

    def predict(self, features: list, rescale_to: Tuple[int, int] = (512, 512)) -> torch.Tensor:
        """
        Inference-time prediction (no dropout, with upsampling).

        Parameters
        ----------
        features : list of Tensor
            Same as forward().
        rescale_to : tuple of int
            Target (H, W) to bilinearly upsample the prediction to.
            Typically the ground truth resolution for computing metrics.

        Returns
        -------
        Tensor of shape (B, num_classes, H_out, W_out)
        """
        if len(features) > 1:
            target_size = features[0].shape[2:]
            features = [
                F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
                for f in features
            ]
        x = torch.cat(features, dim=1)
        x = self.batchnorm(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=rescale_to, mode="bilinear", align_corners=False)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LinearProbeConfig:
    """Configuration for linear probe training."""
    # Dataset
    dataset_name: str = "ade20k"          # "ade20k" or "voc"
    dataset_root: str = "data/ADEChallengeData2016"
    num_classes: int = 150                # ADE20k: 150, VOC: 21

    # Model
    backbone_name: str = "vitl16"
    weights_path: str = "weights/dinov3_vitl16.pth"
    autocast_dtype: str = "float32"       # "float32" or "bfloat16"

    # Training
    batch_size: int = 2
    total_iterations: int = 40_000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    warmup_iterations: int = 1500

    # Evaluation
    eval_interval: int = 5000
    eval_crop_size: int = 512
    eval_stride: int = 341
    img_size: int = 512

    # Output
    output_dir: str = "results/linear_probe"


def train_linear_probe(config: LinearProbeConfig):
    """
    Train a linear segmentation head on frozen DINOv3 features.

    This is the main entry point for reproducing Section 6.1.2 of the paper.
    It follows the exact same procedure as the DINOv3 codebase:
    1. Load and freeze the backbone
    2. Build a LinearSegmentationHead on top
    3. Train with AdamW + OneCycleLR scheduler
    4. Evaluate with sliding window inference at specified intervals
    5. Save the best model checkpoint

    Parameters
    ----------
    config : LinearProbeConfig
        Training configuration (see dataclass above for all options).

    Returns
    -------
    dict
        Final best metrics, e.g. {"mIoU": 54.9, "acc": ..., ...}

    Notes
    -----
    For the official DINOv3 distributed training pipeline, use:
        ``PYTHONPATH=. python -m dinov3.run.submit dinov3/eval/segmentation/run.py ...``
    This function provides a simplified single-GPU version for experimentation.
    """
    os.makedirs(config.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Step 1: Load frozen backbone ──
    from src.model_loading import load_backbone_with_intermediate_layers

    autocast_dtype = (
        torch.bfloat16 if config.autocast_dtype == "bfloat16" else torch.float32
    )

    feature_extractor = load_backbone_with_intermediate_layers(
        model_name=config.backbone_name,
        weights_path=config.weights_path,
        layer_indices=None,  # Last layer only (for linear probing)
        device=str(device),
        autocast_dtype=autocast_dtype,
    )

    # ── Step 2: Build segmentation head ──
    from src.model_loading import get_model_info
    model_info = get_model_info(config.backbone_name)
    embed_dim = model_info["embed_dim"]

    head = LinearSegmentationHead(
        in_channels=[embed_dim],
        num_classes=config.num_classes,
    ).to(device)

    # ── Step 3: Create dataloaders ──
    from src.data_utils import get_segmentation_dataloaders

    train_loader, val_loader = get_segmentation_dataloaders(
        dataset_name=config.dataset_name,
        dataset_root=config.dataset_root,
        img_size=config.img_size,
        batch_size=config.batch_size,
    )

    # ── Step 4: Optimizer & Scheduler ──
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=config.total_iterations,
        pct_start=config.warmup_iterations / config.total_iterations,
        anneal_strategy="cos",
    )
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # ── Step 5: Training loop ──
    logger.info(f"Starting training for {config.total_iterations} iterations")
    head.train()
    best_miou = 0.0
    global_step = 0
    data_iter = iter(train_loader)

    pbar = tqdm(total=config.total_iterations, desc="Training")
    while global_step < config.total_iterations:
        try:
            images, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, targets = next(data_iter)

        images = images.to(device)          # (B, 3, H, W)
        targets = targets.to(device).long()  # (B, H, W)

        # Forward through frozen backbone
        with torch.no_grad():
            features = feature_extractor(images)
            # features: list of (B, embed_dim, H_feat, W_feat)

        # Forward through trainable head
        with torch.autocast("cuda", dtype=autocast_dtype):
            logits = head(features)  # (B, num_classes, H_feat, W_feat)

            # Upsample logits to match ground truth resolution
            if logits.shape[-2:] != targets.shape[-2:]:
                logits = F.interpolate(
                    logits, size=targets.shape[-2:],
                    mode="bilinear", align_corners=False,
                )

            loss = criterion(logits, targets)

        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        global_step += 1
        pbar.update(1)
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.6f}")

        # ── Periodic evaluation ──
        if global_step % config.eval_interval == 0:
            from src.evaluation import evaluate_segmentation

            metrics = evaluate_segmentation(
                feature_extractor=feature_extractor,
                head=head,
                val_loader=val_loader,
                num_classes=config.num_classes,
                device=device,
                crop_size=config.eval_crop_size,
                stride=config.eval_stride,
                autocast_dtype=autocast_dtype,
            )
            miou = metrics["mIoU"]
            logger.info(f"Step {global_step}: mIoU = {miou:.2f}")
            print(f"\n[Step {global_step}] mIoU = {miou:.2f}")

            if miou > best_miou:
                best_miou = miou
                torch.save(
                    {"head_state_dict": head.state_dict(), "step": global_step, "mIoU": miou},
                    os.path.join(config.output_dir, "best_model.pth"),
                )
                logger.info(f"New best mIoU: {miou:.2f}")

            head.train()  # Switch back to training mode

    pbar.close()

    # Save final model
    torch.save(
        {"head_state_dict": head.state_dict(), "step": global_step},
        os.path.join(config.output_dir, "model_final.pth"),
    )
    logger.info(f"Training complete. Best mIoU: {best_miou:.2f}")
    return {"mIoU": best_miou}


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Train linear segmentation probe on DINOv3")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--backbone", type=str, default="vitl16")
    parser.add_argument("--weights", type=str, default="weights/dinov3_vitl16.pth")
    parser.add_argument("--dataset", type=str, default="ade20k")
    parser.add_argument("--dataset-root", type=str, default="data/ADEChallengeData2016")
    parser.add_argument("--output-dir", type=str, default="results/linear_probe")
    parser.add_argument("--iterations", type=int, default=40000)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    num_classes = 150 if args.dataset == "ade20k" else 21

    config = LinearProbeConfig(
        backbone_name=args.backbone,
        weights_path=args.weights,
        dataset_name=args.dataset,
        dataset_root=args.dataset_root,
        num_classes=num_classes,
        output_dir=args.output_dir,
        total_iterations=args.iterations,
        learning_rate=args.lr,
    )
    train_linear_probe(config)
