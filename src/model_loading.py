"""
model_loading.py — Loading Pretrained DINOv3 Backbones
======================================================

This module provides utilities for loading DINOv3 vision transformer backbones
and wrapping them for downstream segmentation tasks.

DINOv3 Architecture Overview
----------------------------
DINOv3 uses a Vision Transformer (ViT) architecture trained with self-supervised
learning (SSL). Key architectural details:

- **Patch Embedding**: Input images are split into 16×16 pixel patches. Each patch
  is linearly projected to the embedding dimension. For a 512×512 image, this
  produces a 32×32 = 1024 patch token sequence.

- **Positional Encoding**: Uses Rotary Position Embeddings (RoPE) instead of
  learnable absolute positions. RoPE encodes relative positions between patches
  in the attention mechanism, enabling resolution flexibility at inference time.

- **Transformer Blocks**: Stack of self-attention + FFN (SwiGLU) blocks.
  ViT-L has 24 blocks, ViT-7B has 40 blocks.

- **Register Tokens**: 4 additional tokens prepended to the sequence that act as
  "scratch space" for the attention mechanism, preventing high-norm patch outliers.

- **CLS Token**: A global class token that aggregates image-level information.

- **Output**: The backbone produces per-patch feature vectors (dense features) and
  a CLS token (global feature). For segmentation, we use the dense patch features.

Available Models
----------------
- ``dinov3_vitl16``  : ViT-Large/16 (300M params, 24 blocks, embed_dim=1024)
- ``dinov3_vit7b16`` : ViT-7B/16 (6.7B params, 40 blocks, embed_dim=4096)

Both are available with weights pretrained on the LVD-1689M web dataset.

Usage Example
-------------
>>> from src.model_loading import load_backbone, load_backbone_with_intermediate_layers
>>>
>>> # Load ViT-L backbone
>>> backbone = load_backbone("vitl16", weights_path="weights/dinov3_vitl16.pth")
>>> # backbone: nn.Module, input: (B, 3, H, W), output: (B, N, 1024)
>>>
>>> # Load with intermediate layer extraction for segmentation
>>> feature_extractor = load_backbone_with_intermediate_layers(
...     "vitl16", weights_path="weights/dinov3_vitl16.pth"
... )
>>> # feature_extractor: input (B, 3, 512, 512) → list of (B, 1024, 32, 32)
"""

import logging
from functools import partial
from typing import Optional, Sequence

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def load_backbone(
    model_name: str = "vitl16",
    weights_path: Optional[str] = None,
    device: str = "cuda",
) -> nn.Module:
    """
    Load a pretrained DINOv3 backbone model.

    This function loads a ViT backbone from the DINOv3 model family. The backbone
    is a standard Vision Transformer that processes images into patch-level features.

    Parameters
    ----------
    model_name : str
        Which backbone variant to load. Options:
        - "vitl16" : ViT-Large with patch size 16 (300M params)
        - "vit7b16" : ViT-7B with patch size 16 (6.7B params)
    weights_path : str, optional
        Path or URL to pretrained weights file (.pth).
        If None, loads with default weights via torch.hub.
    device : str
        Device to load the model onto ("cuda" or "cpu").

    Returns
    -------
    nn.Module
        The loaded backbone model in eval mode.

        For ViT-L/16 with a 512×512 input:
        - Input shape:  (B, 3, 512, 512) — batch of RGB images
        - Output shape: (B, 1025, 1024) — 1024 patch tokens + 1 CLS token,
          each with dimension 1024

        For ViT-7B/16 with a 512×512 input:
        - Input shape:  (B, 3, 512, 512)
        - Output shape: (B, 1025, 4096) — 1024 patch tokens + 1 CLS,
          each with dimension 4096

    Example
    -------
    >>> backbone = load_backbone("vitl16", weights_path="weights/dinov3_vitl16.pth")
    >>> dummy_input = torch.randn(1, 3, 512, 512).cuda()
    >>> with torch.no_grad():
    ...     output = backbone(dummy_input)
    >>> print(output.shape)  # (1, 1025, 1024)
    """
    import dinov3.hub.backbones as hub_backbones

    hub_fn_map = {
        "vitl16": hub_backbones.dinov3_vitl16,
        "vit7b16": hub_backbones.dinov3_vit7b16,
        "vitb16": hub_backbones.dinov3_vitb16,
        "vits16": hub_backbones.dinov3_vits16,
    }

    if model_name not in hub_fn_map:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(hub_fn_map.keys())}"
        )

    logger.info(f"Loading DINOv3 backbone: {model_name}")

    if weights_path:
        backbone = hub_fn_map[model_name](pretrained=True, weights=weights_path)
    else:
        backbone = hub_fn_map[model_name](pretrained=True)

    backbone = backbone.to(device).eval()
    n_params = sum(p.numel() for p in backbone.parameters())
    logger.info(f"Loaded {model_name} with {n_params / 1e6:.1f}M parameters")

    return backbone


def load_backbone_with_intermediate_layers(
    model_name: str = "vitl16",
    weights_path: Optional[str] = None,
    layer_indices: Optional[Sequence[int]] = None,
    device: str = "cuda",
    autocast_dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """
    Load a DINOv3 backbone wrapped to extract intermediate layer features.

    This is the key function for segmentation tasks. Instead of only getting
    the final layer output, this wrapper extracts features from specific
    intermediate transformer blocks. These multi-scale features capture both
    low-level spatial details (early layers) and high-level semantics (later layers).

    For the **linear probe** (Section 6.1.2), only the LAST layer is used.
    For the **Mask2Former** (Section 6.3.2), four evenly-spaced layers are used.

    Parameters
    ----------
    model_name : str
        Backbone variant ("vitl16" or "vit7b16").
    weights_path : str, optional
        Path or URL to pretrained weights.
    layer_indices : list of int, optional
        Which transformer block indices to extract features from.
        If None, defaults to [last_layer] (for linear probing).
        For ViT-L (24 blocks): [4, 11, 17, 23] extracts 4 evenly-spaced layers.
        For ViT-7B (40 blocks): [9, 19, 29, 39] extracts 4 evenly-spaced layers.
    device : str
        Device to load onto.
    autocast_dtype : torch.dtype
        Precision for autocast during forward pass. Use torch.bfloat16 for
        memory efficiency on H100/A100, or torch.float32 for full precision.

    Returns
    -------
    nn.Module
        A ``ModelWithIntermediateLayers`` wrapper.

        For ViT-L/16 extracting the last layer, with 512×512 input:
        - Input:  (B, 3, 512, 512)
        - Output: list of 1 tensor, each (B, 1024, 32, 32)
          [1024 = embed_dim, 32×32 = spatial grid of patch features]

        For ViT-L/16 extracting 4 layers, with 512×512 input:
        - Input:  (B, 3, 512, 512)
        - Output: list of 4 tensors, each (B, 1024, 32, 32)

    Example
    -------
    >>> # For linear probing (last layer only)
    >>> extractor = load_backbone_with_intermediate_layers(
    ...     "vitl16", weights_path="weights/dinov3_vitl16.pth"
    ... )
    >>> images = torch.randn(2, 3, 512, 512).cuda()
    >>> features = extractor(images)
    >>> print(len(features), features[0].shape)
    1 torch.Size([2, 1024, 32, 32])

    >>> # For Mask2Former (4 intermediate layers)
    >>> extractor = load_backbone_with_intermediate_layers(
    ...     "vitl16",
    ...     weights_path="weights/dinov3_vitl16.pth",
    ...     layer_indices=[4, 11, 17, 23],
    ... )
    >>> features = extractor(images)
    >>> print(len(features))  # 4
    """
    from dinov3.eval.utils import ModelWithIntermediateLayers

    backbone = load_backbone(model_name, weights_path, device)

    # Default: extract only the last layer (for linear probing)
    if layer_indices is None:
        n_blocks = backbone.n_blocks
        layer_indices = [n_blocks - 1]

    autocast_ctx = partial(
        torch.autocast, device_type="cuda", enabled=True, dtype=autocast_dtype
    )

    feature_extractor = ModelWithIntermediateLayers(
        feature_model=backbone,
        n=layer_indices,
        autocast_ctx=autocast_ctx,
        reshape=True,            # Reshape flat tokens to 2D spatial grid
        return_class_token=False,  # We don't need CLS for segmentation
    )
    feature_extractor.requires_grad_(False)  # Freeze backbone

    logger.info(
        f"Feature extractor ready. Extracting layers {layer_indices} "
        f"from {model_name} (frozen, autocast={autocast_dtype})"
    )

    return feature_extractor


def get_model_info(model_name: str) -> dict:
    """
    Return architecture metadata for a given DINOv3 model variant.

    Parameters
    ----------
    model_name : str
        One of "vits16", "vitb16", "vitl16", "vit7b16".

    Returns
    -------
    dict
        Dictionary with keys: embed_dim, n_blocks, n_heads, patch_size, n_params.

    Example
    -------
    >>> info = get_model_info("vitl16")
    >>> print(info)
    {'embed_dim': 1024, 'n_blocks': 24, 'n_heads': 16, 'patch_size': 16, 'n_params': '300M'}
    """
    model_specs = {
        "vits16": dict(embed_dim=384, n_blocks=12, n_heads=6, patch_size=16, n_params="21M"),
        "vitb16": dict(embed_dim=768, n_blocks=12, n_heads=12, patch_size=16, n_params="86M"),
        "vitl16": dict(embed_dim=1024, n_blocks=24, n_heads=16, patch_size=16, n_params="300M"),
        "vit7b16": dict(embed_dim=4096, n_blocks=40, n_heads=32, patch_size=16, n_params="6,716M"),
    }
    if model_name not in model_specs:
        raise ValueError(f"Unknown model. Available: {list(model_specs.keys())}")
    return model_specs[model_name]
