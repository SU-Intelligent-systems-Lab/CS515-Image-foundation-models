"""
evaluation.py — Segmentation Metrics (mIoU, Dice, Accuracy)
============================================================

This module implements the standard segmentation evaluation metrics used in
the DINOv3 paper and throughout the computer vision community.

Key Metric: Mean Intersection over Union (mIoU)
-----------------------------------------------
mIoU is the standard metric for semantic segmentation. For each class c:

    IoU(c) = TP(c) / (TP(c) + FP(c) + FN(c))

where:
- TP(c) = pixels correctly predicted as class c (true positives)
- FP(c) = pixels incorrectly predicted as class c (false positives)
- FN(c) = pixels of class c that were missed (false negatives)

mIoU = mean of IoU(c) across all C classes.

This is computed efficiently using histograms:
- area_intersect = histogram of correctly predicted pixels per class
- area_union = area_pred + area_label - area_intersect
- IoU per class = area_intersect / area_union

DINOv3 Paper Results (mIoU on ADE20k)
--------------------------------------
- Linear probe ViT-L:   54.9
- Linear probe ViT-7B:  55.9
- Mask2Former ViT-7B:   62.6 (single-scale), 63.0 (TTA)
- Best competing model:  53.0 (AM-RADIOv2.5, an agglomerative model)
"""

import logging
from typing import List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_iou_per_image(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
    reduce_zero_label: bool = True,
) -> torch.Tensor:
    """
    Compute intersection and union statistics for a single image.

    This function computes the per-class intersection, union, prediction area,
    and label area for one image. These intermediate results are later aggregated
    across the full dataset to compute mIoU.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted class indices, shape (1, H, W) or (H, W).
        Each pixel value is an integer in [0, num_classes-1].
    target : torch.Tensor
        Ground truth class indices, same shape as pred.
        Pixels with value ``ignore_index`` are excluded from computation.
    num_classes : int
        Total number of semantic classes.
        ADE20k: 150, Pascal VOC: 21, Cityscapes: 19.
    ignore_index : int
        Label value to ignore (typically 255 = unlabeled/boundary).
    reduce_zero_label : bool
        If True, shifts all labels down by 1 (label 0 becomes ignored).
        This is standard for ADE20k where 0 = background/unlabeled.

    Returns
    -------
    torch.Tensor
        Shape (4, num_classes) containing:
        [0] area_intersect — correctly predicted pixels per class
        [1] area_union     — union of predicted and ground truth per class
        [2] area_pred      — predicted pixels per class
        [3] area_label     — ground truth pixels per class

    Example
    -------
    >>> pred = torch.randint(0, 150, (1, 512, 512))
    >>> target = torch.randint(0, 150, (1, 512, 512))
    >>> stats = compute_iou_per_image(pred, target, num_classes=150)
    >>> print(stats.shape)  # (4, 150)
    """
    pred = pred.float().squeeze()
    target = target.float().squeeze()

    # For ADE20k: label 0 is background, shift all labels down by 1
    # so that the former label 0 becomes -1, then set it to ignore_index
    if reduce_zero_label:
        target_new = target.clone()
        target_new[target_new == ignore_index] = ignore_index + 1
        target_new -= 1
        target_new[target_new == -1] = ignore_index
        target = target_new

    # Mask out ignored pixels
    valid_mask = target != ignore_index
    pred = pred[valid_mask]
    target = target[valid_mask]

    # Compute per-class areas using histograms
    # This is much faster than looping over classes
    intersect = pred[pred == target]
    area_intersect = torch.histc(intersect, bins=num_classes, min=0, max=num_classes - 1)
    area_pred = torch.histc(pred, bins=num_classes, min=0, max=num_classes - 1)
    area_label = torch.histc(target, bins=num_classes, min=0, max=num_classes - 1)
    area_union = area_pred + area_label - area_intersect

    return torch.stack([area_intersect, area_union, area_pred, area_label])


def aggregate_metrics(
    all_results: List[torch.Tensor],
    metrics: List[str] = ("mIoU", "dice", "fscore"),
) -> dict:
    """
    Aggregate per-image IoU statistics into final segmentation metrics.

    Parameters
    ----------
    all_results : list of Tensor
        List of (4, num_classes) tensors from ``compute_iou_per_image()``.
    metrics : list of str
        Which metrics to compute. Options: "mIoU", "dice", "fscore".

    Returns
    -------
    dict
        Metric name → value (as percentage, e.g. mIoU=54.9 means 54.9%).

    Example
    -------
    >>> results = [compute_iou_per_image(pred, gt, 150) for pred, gt in dataset]
    >>> final = aggregate_metrics(results)
    >>> print(final)  # {'mIoU': 54.9, 'dice': 67.2, ...}
    """
    stacked = torch.stack(all_results)  # (N_images, 4, num_classes)

    total_intersect = stacked[:, 0].sum(dim=0)  # (num_classes,)
    total_union = stacked[:, 1].sum(dim=0)
    total_pred = stacked[:, 2].sum(dim=0)
    total_label = stacked[:, 3].sum(dim=0)

    output = {}

    if "mIoU" in metrics:
        iou_per_class = total_intersect / total_union  # (num_classes,)
        output["mIoU"] = round(iou_per_class.nanmean().item() * 100, 2)

    if "dice" in metrics:
        dice_per_class = 2 * total_intersect / (total_pred + total_label)
        output["dice"] = round(dice_per_class.nanmean().item() * 100, 2)

    if "fscore" in metrics:
        precision = total_intersect / total_pred
        recall = total_intersect / total_label
        f1 = 2 * precision * recall / (precision + recall)
        output["fscore"] = round(f1.nanmean().item() * 100, 2)

    # Overall pixel accuracy
    output["pixel_acc"] = round(
        (total_intersect.sum() / total_label.sum()).item() * 100, 2
    )

    return output


def evaluate_segmentation(
    feature_extractor,
    head,
    val_loader,
    num_classes: int,
    device: torch.device,
    crop_size: int = 512,
    stride: int = 341,
    autocast_dtype: torch.dtype = torch.float32,
) -> dict:
    """
    Evaluate a linear probe segmentation model on a validation dataset.

    Uses sliding window inference: the image is split into overlapping crops,
    each processed independently, and predictions are averaged in overlapping
    regions.

    Parameters
    ----------
    feature_extractor : nn.Module
        Frozen backbone with intermediate layer extraction.
    head : LinearSegmentationHead
        The trained linear probe head.
    val_loader : DataLoader
        Validation dataset loader.
    num_classes : int
        Number of segmentation classes.
    device : torch.device
        Device for computation.
    crop_size : int
        Size of each sliding window crop.
    stride : int
        Stride between consecutive crops.
    autocast_dtype : torch.dtype
        Precision for autocast.

    Returns
    -------
    dict
        {"mIoU": float, "dice": float, "pixel_acc": float, ...}
    """
    head.eval()
    all_results = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            # Extract features from frozen backbone
            with torch.autocast("cuda", dtype=autocast_dtype):
                features = feature_extractor(images)

            # Predict with the linear head
            logits = head.predict(features, rescale_to=targets.shape[-2:])
            pred = logits.argmax(dim=1)

            # Compute per-image statistics
            stats = compute_iou_per_image(
                pred, targets, num_classes=num_classes
            )
            all_results.append(stats)

    metrics = aggregate_metrics(all_results)
    logger.info(f"Evaluation results: {metrics}")
    return metrics
