"""
Visualization helpers used by notebooks and analysis scripts.

Functions
---------
attention_heatmap
    Extract and reshape the final-block [CLS]->patch attention of a single
    image into per-head spatial maps.
overlay_heatmap
    Blend a heatmap over an RGB image for visualization.
plot_loss_curves
    Plot loss / grad-norm / LR curves over training steps.
save_figure
    Save a matplotlib figure to ``results/figures/`` with tight layout.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Attention visualization
# -----------------------------------------------------------------------------

@torch.no_grad()
def attention_heatmap(
    attention_weights: torch.Tensor,
    grid_size: int,
) -> np.ndarray:
    """Convert final-block attention weights to per-head spatial heatmaps.

    Parameters
    ----------
    attention_weights : Tensor, shape (B, H, N+1, N+1) or (H, N+1, N+1)
        Self-attention weights from a ViT block. If B > 1, only the first
        sample is used. N+1 is the number of tokens including CLS.
    grid_size : int
        Side length of the patch grid (N = grid_size ** 2).

    Returns
    -------
    heatmaps : np.ndarray of shape (n_heads, grid_size, grid_size)
    """
    if attention_weights.ndim == 4:
        attn = attention_weights[0]                       # (H, N+1, N+1)
    else:
        attn = attention_weights
    # Attention from CLS token to every patch token
    cls_attn = attn[:, 0, 1:]                             # (H, N)
    n_heads, n_tokens = cls_attn.shape
    assert n_tokens == grid_size * grid_size, (
        f"grid_size**2={grid_size**2} does not match n_tokens={n_tokens}"
    )
    maps = cls_attn.reshape(n_heads, grid_size, grid_size).cpu().numpy()
    return maps


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    cmap: str = "jet",
) -> np.ndarray:
    """Blend a heatmap over an RGB image.

    Parameters
    ----------
    image : np.ndarray, shape (H, W, 3), uint8 or float in [0, 1]
    heatmap : np.ndarray, shape (H', W') -- will be resized to (H, W)
    alpha : float
        Opacity of the heatmap.
    cmap : str
        matplotlib colormap name.

    Returns
    -------
    blended : np.ndarray, shape (H, W, 3), float in [0, 1]
    """
    import matplotlib.cm as cm

    if image.dtype == np.uint8:
        img_f = image.astype(np.float32) / 255.0
    else:
        img_f = image.astype(np.float32)
        if img_f.max() > 1.0:
            img_f = img_f / 255.0

    H, W = img_f.shape[:2]
    # Resize heatmap with bilinear upsampling (via torch for convenience)
    h = torch.from_numpy(heatmap).float()[None, None]            # (1,1,h,w)
    h = F.interpolate(h, size=(H, W), mode="bilinear", align_corners=False)
    h = h[0, 0].numpy()
    # Normalize heatmap to [0, 1]
    h_min, h_max = float(h.min()), float(h.max())
    if h_max - h_min > 1e-8:
        h = (h - h_min) / (h_max - h_min)
    else:
        h = np.zeros_like(h)

    colormap = cm.get_cmap(cmap)
    colored = colormap(h)[..., :3]                               # (H, W, 3)
    blended = (1 - alpha) * img_f + alpha * colored
    return np.clip(blended, 0, 1)


# -----------------------------------------------------------------------------
# Training curve plots
# -----------------------------------------------------------------------------

def plot_loss_curves(
    history: Dict[str, List[float]],
    out_path: str | os.PathLike,
    title: str = "Training curves",
) -> None:
    """Plot every scalar series in ``history`` on a grid of subplots.

    ``history`` is typically ``{"train/loss": [...], "train/lr": [...], ...}``.
    """
    import matplotlib.pyplot as plt

    keys = list(history.keys())
    n = len(keys)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)

    for i, k in enumerate(keys):
        ax = axes[i // cols][i % cols]
        ax.plot(history[k], linewidth=1.0)
        ax.set_title(k)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)

    # Hide any unused axes
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle(title)
    save_figure(fig, out_path)


def save_figure(fig, out_path: str | os.PathLike, dpi: int = 150) -> None:
    """Save a matplotlib figure with tight layout, creating parents as needed."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Denormalization helper
# -----------------------------------------------------------------------------

def denormalize(
    img_tensor: torch.Tensor,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Convert a normalized CHW tensor back to HWC float [0, 1] image.

    Accepts a single tensor (C, H, W) or a batch and returns the first element.
    """
    if img_tensor.ndim == 4:
        img_tensor = img_tensor[0]
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    out = img_tensor.detach().cpu() * std_t + mean_t
    out = out.clamp(0, 1).permute(1, 2, 0).numpy()
    return out
