"""
visualization.py — Feature Visualization and Segmentation Overlays
==================================================================

This module provides visualization utilities for:
1. **PCA Feature Maps**: Project high-dimensional DINOv3 patch features into
   3D using PCA, then map to RGB for visualization. This reveals the semantic
   structure learned by the backbone (see Figures 4 and 13 in the paper).
2. **Segmentation Overlays**: Overlay predicted segmentation masks on input
   images for qualitative evaluation.
3. **Comparison Grids**: Side-by-side comparison of different models/methods.
"""

import logging
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


def compute_pca_features(
    features: torch.Tensor,
    n_components: int = 3,
    background_threshold: Optional[float] = None,
) -> np.ndarray:
    """
    Project patch features to 3 dimensions using PCA for RGB visualization.

    This reproduces the PCA visualizations shown in Figures 4, 13, and 17
    of the DINOv3 paper. The first 3 principal components are mapped to
    RGB channels, revealing semantic groupings in the feature space.

    Parameters
    ----------
    features : torch.Tensor
        Dense patch features of shape (C, H, W) or (1, C, H, W).
        - C = embedding dimension (e.g. 1024 for ViT-L, 4096 for ViT-7B)
        - H, W = spatial dimensions of the feature grid
        For ViT-L/16 with 512×512 input: (1024, 32, 32)
    n_components : int
        Number of PCA components (typically 3 for RGB mapping).
    background_threshold : float, optional
        If provided, pixels whose first PCA component is below this threshold
        are treated as background and masked out (set to white).
        The paper uses this to "focus the PCA on the subject" (Figure 4).

    Returns
    -------
    np.ndarray
        RGB image of shape (H, W, 3) with values in [0, 1].

    Example
    -------
    >>> backbone = load_backbone_with_intermediate_layers("vitl16", ...)
    >>> features = backbone(image)[0]  # (1, 1024, 32, 32)
    >>> pca_rgb = compute_pca_features(features.squeeze(0))  # (32, 32, 3)
    >>> plt.imshow(pca_rgb)
    """
    if features.dim() == 4:
        features = features.squeeze(0)

    C, H, W = features.shape
    # Reshape to (N_patches, C) for PCA
    feat_flat = features.reshape(C, -1).T.float().cpu()  # (H*W, C)

    # Center the data
    mean = feat_flat.mean(dim=0, keepdim=True)
    feat_centered = feat_flat - mean

    # PCA via SVD (more numerically stable than eigendecomposition)
    U, S, Vt = torch.linalg.svd(feat_centered, full_matrices=False)
    components = U[:, :n_components]  # (H*W, n_components)

    # Normalize each component to [0, 1] for visualization
    for i in range(n_components):
        col = components[:, i]
        components[:, i] = (col - col.min()) / (col.max() - col.min() + 1e-8)

    pca_rgb = components.reshape(H, W, n_components).numpy()

    # Optional background masking
    if background_threshold is not None:
        # Use the first component as a foreground indicator
        fg_mask = pca_rgb[:, :, 0] > background_threshold
        pca_rgb[~fg_mask] = 1.0  # White background

    return pca_rgb


def visualize_segmentation(
    image: Image.Image,
    pred_mask: np.ndarray,
    gt_mask: Optional[np.ndarray] = None,
    alpha: float = 0.5,
    num_classes: int = 150,
    title: str = "",
    save_path: Optional[str] = None,
):
    """
    Visualize segmentation predictions overlaid on the input image.

    Parameters
    ----------
    image : PIL.Image
        Original input image.
    pred_mask : np.ndarray
        Predicted class indices, shape (H, W), values in [0, num_classes-1].
    gt_mask : np.ndarray, optional
        Ground truth mask for comparison. Same shape as pred_mask.
    alpha : float
        Transparency of the overlay (0 = fully transparent, 1 = fully opaque).
    num_classes : int
        Total number of classes (for colormap).
    title : str
        Plot title.
    save_path : str, optional
        If provided, save the figure to this path.

    Example
    -------
    >>> visualize_segmentation(image, pred_mask, gt_mask, title="ADE20k Prediction")
    """
    # Generate a colormap
    cmap = plt.cm.get_cmap("Spectral", num_classes)

    n_cols = 3 if gt_mask is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Prediction overlay
    axes[1].imshow(image)
    axes[1].imshow(cmap(pred_mask / num_classes), alpha=alpha)
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    # Ground truth (if available)
    if gt_mask is not None:
        axes[2].imshow(image)
        axes[2].imshow(cmap(gt_mask / num_classes), alpha=alpha)
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved visualization to {save_path}")

    plt.show()


def visualize_pca_comparison(
    image: Image.Image,
    feature_maps: dict,
    figsize: Tuple[int, int] = (20, 5),
    save_path: Optional[str] = None,
):
    """
    Compare PCA feature visualizations across different models.

    Reproduces Figure 13 from the paper, comparing the dense feature quality
    of DINOv3 against other backbones (SigLIP 2, PEspatial, DINOv2).

    Parameters
    ----------
    image : PIL.Image
        Original input image.
    feature_maps : dict
        Mapping from model name to feature tensor.
        Example: {"DINOv3 ViT-L": features_v3, "DINOv2 ViT-g": features_v2}
        Each tensor: (C, H, W) or (1, C, H, W).
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save the figure.

    Example
    -------
    >>> visualize_pca_comparison(
    ...     image,
    ...     {"DINOv3": features_v3, "DINOv2": features_v2},
    ... )
    """
    n_models = len(feature_maps)
    fig, axes = plt.subplots(1, n_models + 1, figsize=figsize)

    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[0].axis("off")

    for i, (name, features) in enumerate(feature_maps.items()):
        pca_rgb = compute_pca_features(features)
        # Resize PCA map to image size for better visualization
        pca_resized = np.array(
            Image.fromarray((pca_rgb * 255).astype(np.uint8)).resize(
                image.size, resample=Image.BILINEAR
            )
        )
        axes[i + 1].imshow(pca_resized)
        axes[i + 1].set_title(name)
        axes[i + 1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_curves(
    log_file: str,
    save_path: Optional[str] = None,
):
    """
    Plot training loss and validation mIoU curves from a training log.

    Parameters
    ----------
    log_file : str
        Path to a CSV or JSON log file containing 'step', 'loss', 'mIoU' columns.
    save_path : str, optional
        Path to save the figure.
    """
    import pandas as pd

    df = pd.read_csv(log_file)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    if "loss" in df.columns:
        ax1.plot(df["step"], df["loss"])
        ax1.set_xlabel("Training Step")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss")

    if "mIoU" in df.columns:
        eval_df = df[df["mIoU"].notna()]
        ax2.plot(eval_df["step"], eval_df["mIoU"], "o-")
        ax2.set_xlabel("Training Step")
        ax2.set_ylabel("mIoU (%)")
        ax2.set_title("Validation mIoU")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
