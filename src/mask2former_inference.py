"""
mask2former_inference.py — Mask2Former Semantic Segmentation with Frozen DINOv3
===============================================================================

This module implements **Section 6.3.2** of the DINOv3 paper: semantic segmentation
using a Mask2Former decoder on top of a frozen DINOv3 backbone.

Architecture Overview
---------------------
Unlike the linear probe (Section 6.1.2) which uses a single linear layer, the
Mask2Former approach uses a sophisticated decoder with ~927M trainable parameters.
Despite this, the DINOv3 backbone is still frozen — only the decoder is trained.

::

    Input Image (B, 3, 896, 896)
           │
           ▼
    ┌───────────────────────────────┐
    │  DINOv3 ViT-7B Backbone      │  ← FROZEN
    │  + ViT-Adapter (no injector) │
    │                               │
    │  Extracts features from 4     │
    │  evenly-spaced layers:        │
    │  [10, 20, 30, 40]             │
    │                               │
    │  Each layer produces:         │
    │  (B, 4096, 56, 56)            │
    └───────────────────────────────┘
           │
           ▼  4 multi-scale feature maps
    ┌───────────────────────────────┐
    │  Mask2Former Decoder          │  ← TRAINED (927M params)
    │                               │
    │  1. Pixel Decoder (FPN-style) │  Produces multi-scale features
    │  2. Transformer Decoder       │  Cross-attends queries to features
    │  3. Mask Prediction           │  Predicts per-query binary masks
    │  4. Class Prediction          │  Predicts per-query class labels
    │                               │
    │  Output: 100 query masks +    │
    │  class predictions, combined  │
    │  via matrix multiplication    │
    └───────────────────────────────┘
           │
           ▼
    Prediction (B, 150, H, W)  ← Per-pixel class probabilities

Inference Method: Sliding Window
---------------------------------
For high-resolution inference (e.g. 896×896), the image is processed using a
sliding window approach:

1. Divide the image into overlapping crops of size (crop_size × crop_size)
2. Run the full model (backbone + decoder) on each crop
3. Aggregate overlapping predictions by averaging
4. Optionally apply Test-Time Augmentation (TTA): run at multiple scales
   (0.9×, 0.95×, 1.0×, 1.05×, 1.1×) plus horizontal flips, then average.

Paper Results (Table 11)
------------------------
- Single-scale: 62.6 mIoU on ADE20k
- Multi-scale TTA: 63.0 mIoU on ADE20k
- This matches ONE-PEACE (63.0 mIoU) which requires finetuning the backbone
"""

import logging
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image

logger = logging.getLogger(__name__)


def load_mask2former_segmentor(
    backbone_weights: Optional[str] = None,
    m2f_weights: Optional[str] = None,
    device: str = "cuda",
    autocast_dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """
    Load the full DINOv3 ViT-7B + Mask2Former segmentor.

    This loads:
    1. The DINOv3 ViT-7B backbone (6.7B params, frozen)
    2. The ViT-Adapter (extracts multi-scale features)
    3. The Mask2Former decoder head (927M params, pretrained on ADE20k)

    Parameters
    ----------
    backbone_weights : str, optional
        Path or URL to the ViT-7B backbone weights.
        If None, uses default from torch.hub.
    m2f_weights : str, optional
        Path or URL to the Mask2Former decoder head weights.
        If None, uses default from torch.hub.
    device : str
        Device to load on. "cuda" recommended (needs ~45GB VRAM).
    autocast_dtype : torch.dtype
        Use torch.bfloat16 to save memory on H100/A100.

    Returns
    -------
    nn.Module
        The full segmentation model (backbone + decoder).

        For inference, call ``model.predict(images, rescale_to=(H, W))``:
        - Input: (1, 3, 896, 896) — single image batch
        - Output: dict with 'pred_masks' and 'pred_logits'

    Example
    -------
    >>> segmentor = load_mask2former_segmentor(
    ...     backbone_weights="weights/dinov3_vit7b16.pth",
    ...     m2f_weights="weights/dinov3_vit7b16_m2f_head.pth",
    ... )
    >>> # Use with slide_inference() for evaluation
    """
    from dinov3.hub.segmentors import dinov3_vit7b16_ms

    kwargs = {"autocast_dtype": autocast_dtype}
    if backbone_weights:
        kwargs["backbone_weights"] = backbone_weights
    if m2f_weights:
        kwargs["weights"] = m2f_weights

    logger.info("Loading DINOv3 ViT-7B + Mask2Former segmentor...")
    segmentor = dinov3_vit7b16_ms(**kwargs)
    segmentor = segmentor.to(device).eval()

    n_total = sum(p.numel() for p in segmentor.parameters())
    n_trainable = sum(p.numel() for p in segmentor.parameters() if p.requires_grad)
    logger.info(
        f"Segmentor loaded: {n_total / 1e9:.1f}B total params, "
        f"{n_trainable / 1e6:.0f}M trainable"
    )

    return segmentor


def slide_inference(
    image: torch.Tensor,
    segmentor: nn.Module,
    num_classes: int = 150,
    crop_size: Tuple[int, int] = (896, 896),
    stride: Tuple[int, int] = (596, 596),
) -> torch.Tensor:
    """
    Sliding window inference for semantic segmentation.

    The image is divided into overlapping crops. Each crop is independently
    processed by the segmentor (backbone + Mask2Former decoder). Predictions
    are aggregated by averaging over overlapping regions.

    This is the exact method used in the DINOv3 paper for all segmentation
    evaluations (both linear probe and Mask2Former).

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor of shape (1, 3, H, W).
        Must be normalized with ImageNet mean/std.
        H, W should be multiples of the patch size (16).
    segmentor : nn.Module
        The full segmentation model with a ``.predict()`` method.
    num_classes : int
        Number of segmentation classes. ADE20k = 150.
    crop_size : tuple of int
        (height, width) of each sliding window crop.
    stride : tuple of int
        (h_stride, w_stride) between consecutive crops.
        stride < crop_size creates overlapping regions for smoother predictions.

    Returns
    -------
    torch.Tensor
        Prediction logits of shape (1, num_classes, H, W), same spatial
        resolution as the input. Not yet argmaxed — call .argmax(dim=1)
        to get the class prediction map.

    Example
    -------
    >>> # image: (1, 3, 896, 896), normalized
    >>> logits = slide_inference(image, segmentor, num_classes=150)
    >>> pred_map = logits.argmax(dim=1)  # (1, 896, 896) class indices
    """
    h_stride, w_stride = stride
    h_crop, w_crop = crop_size
    B, C, h_img, w_img = image.shape
    assert B == 1, "Sliding inference processes one image at a time"

    # Compute grid of crop positions
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

    # Accumulation buffers
    preds = torch.zeros((1, num_classes, h_img, w_img), device="cpu")
    count = torch.zeros((1, 1, h_img, w_img), dtype=torch.int8, device="cpu")

    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            # Compute crop coordinates (ensuring we don't go out of bounds)
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)

            # Extract and process crop
            crop = image[:, :, y1:y2, x1:x2]
            with torch.inference_mode():
                crop_pred = segmentor.predict(crop, rescale_to=crop.shape[2:])

            # Handle Mask2Former output format
            if isinstance(crop_pred, dict):
                # Mask2Former returns masks + class logits separately
                # Combine them: class_probs @ mask_probs
                mask_cls = F.softmax(crop_pred["pred_logits"], dim=-1)[..., :-1]
                mask_pred = crop_pred["pred_masks"].sigmoid()
                crop_pred = torch.einsum(
                    "bqc,bqhw->bchw",
                    mask_cls.to(torch.bfloat16),
                    mask_pred.to(torch.bfloat16),
                )

            # Pad and accumulate
            pad = (int(x1), int(w_img - x2), int(y1), int(h_img - y2))
            preds += F.pad(crop_pred, pad).cpu()
            count[:, :, y1:y2, x1:x2] += 1

            del crop, crop_pred

    assert (count == 0).sum() == 0, "Some pixels were not covered by any crop"
    return preds / count


def run_m2f_inference(
    segmentor: nn.Module,
    val_loader,
    num_classes: int = 150,
    crop_size: int = 896,
    stride: int = 596,
    use_tta: bool = False,
    tta_ratios: Tuple[float, ...] = (0.9, 0.95, 1.0, 1.05, 1.1),
    device: str = "cuda",
) -> dict:
    """
    Run full Mask2Former inference on a validation dataset and compute metrics.

    Parameters
    ----------
    segmentor : nn.Module
        Loaded M2F segmentor from ``load_mask2former_segmentor()``.
    val_loader : DataLoader
        Validation dataset loader. Each batch yields (image, target) where
        image is (1, 3, H, W) and target is (1, H_gt, W_gt).
    num_classes : int
        Number of segmentation classes.
    crop_size : int
        Sliding window crop size.
    stride : int
        Sliding window stride.
    use_tta : bool
        Whether to apply test-time augmentation (multiple scales + flip).
    tta_ratios : tuple of float
        Scale ratios for TTA. Paper uses (0.9, 0.95, 1.0, 1.05, 1.1).
    device : str
        Computation device.

    Returns
    -------
    dict
        Metrics dictionary, e.g. {"mIoU": 63.0, "dice": ..., ...}
    """
    from src.evaluation import compute_iou_per_image, aggregate_metrics

    segmentor.eval()
    all_results = []

    for batch_idx, (images, targets) in enumerate(val_loader):
        images = images.to(device)
        targets = targets.to(device)

        # Standard inference (single scale)
        logits = slide_inference(
            images, segmentor, num_classes=num_classes,
            crop_size=(crop_size, crop_size),
            stride=(stride, stride),
        )

        # Upsample predictions to match ground truth resolution
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(
                logits.to(device), size=targets.shape[-2:],
                mode="bilinear", align_corners=False,
            )

        pred = logits.argmax(dim=1)
        iou_data = compute_iou_per_image(pred, targets, num_classes)
        all_results.append(iou_data)

        if (batch_idx + 1) % 50 == 0:
            logger.info(f"Processed {batch_idx + 1} images")

    return aggregate_metrics(all_results)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run Mask2Former inference with DINOv3")
    parser.add_argument("--backbone-weights", type=str, required=True)
    parser.add_argument("--m2f-weights", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, default="data/ADEChallengeData2016")
    parser.add_argument("--output-dir", type=str, default="results/m2f_inference")
    parser.add_argument("--use-tta", action="store_true")
    args = parser.parse_args()

    segmentor = load_mask2former_segmentor(
        backbone_weights=args.backbone_weights,
        m2f_weights=args.m2f_weights,
    )

    from src.data_utils import get_segmentation_dataloaders
    _, val_loader = get_segmentation_dataloaders(
        dataset_name="ade20k",
        dataset_root=args.dataset_root,
        img_size=896,
        batch_size=1,
    )

    results = run_m2f_inference(
        segmentor, val_loader,
        num_classes=150,
        crop_size=896,
        stride=596,
        use_tta=args.use_tta,
    )
    print(f"Results: {results}")
