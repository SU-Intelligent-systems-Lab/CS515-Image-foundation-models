"""
Vision Transformer backbone and DINO projection head.

This module contains a clean, pedagogical implementation of a Vision
Transformer (ViT) along with the projection head described in the DINO paper
(Caron et al., 2021, Sec. 3). For the project's mini-reimplementation we use a
ViT-Tiny / ViT-Small configuration so that CIFAR-style training is tractable
on a single consumer GPU. The interface deliberately mirrors `timm` so that a
pretrained backbone can be swapped in for downstream experiments.

Notation
--------
B  : batch size
N  : number of patch tokens per image (H * W / patch_size**2)
d  : token embedding dimension
K  : DINO head output dimension (the "prototype" space)
L  : number of Transformer blocks
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Configuration dataclasses
# -----------------------------------------------------------------------------

@dataclass
class ViTConfig:
    """Configuration for the Vision Transformer backbone."""

    image_size: int = 96              # input side length in pixels (square)
    patch_size: int = 8               # patch side length (image_size % patch_size == 0)
    in_channels: int = 3
    embed_dim: int = 192              # d
    depth: int = 12                   # L
    num_heads: int = 3
    mlp_ratio: float = 4.0            # FFN hidden_dim = embed_dim * mlp_ratio
    qkv_bias: bool = True
    drop_rate: float = 0.0            # dropout on MLP output and after attn
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0       # stochastic depth
    layer_norm_eps: float = 1e-6

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2


# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Convert an image into a sequence of patch embeddings via a Conv2d.

    Equivalent to splitting the image into non-overlapping patches, flattening
    each, and applying a learned linear projection.
    """

    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        assert image_size % patch_size == 0, (
            f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, embed_dim, H/P, W/P) -> (B, N, embed_dim)
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)   # (B, N, embed_dim)
        return x


class DropPath(nn.Module):
    """Stochastic depth per sample (Huang et al., 2016)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # broadcast per sample
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class MLP(nn.Module):
    """Position-wise feed-forward network used inside a Transformer block."""

    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)        # (3, B, H, N, D)
        q, k, v = qkv.unbind(0)                 # each (B, H, N, D)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attn:
            return x, attn
        return x


class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block (standard ViT formulation)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.attn = MultiHeadSelfAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> torch.Tensor:
        if return_attn:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attn=True)
            x = x + self.drop_path(attn_out)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attn_weights
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# -----------------------------------------------------------------------------
# Vision Transformer
# -----------------------------------------------------------------------------

class VisionTransformer(nn.Module):
    """A standard Vision Transformer (ViT) backbone.

    Implements the architecture of Dosovitskiy et al. (2021). The only
    DINO-specific aspect here is that the model optionally supports inputs of
    **variable spatial size** via bicubic interpolation of the positional
    embeddings, which is required by DINO's multi-crop augmentation (global
    crops and smaller local crops are passed through the same backbone).

    Parameters
    ----------
    config : ViTConfig
        Backbone configuration. See `ViTConfig` docstring for defaults.

    Notes
    -----
    Following DINO, we do not use a classification head on the backbone: the
    raw [CLS] embedding is returned and the DINO projection head is applied
    separately (see `DINOHead`).
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config

        self.patch_embed = PatchEmbed(
            image_size=config.image_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )

        # Learnable CLS token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + config.num_patches, config.embed_dim))
        self.pos_drop = nn.Dropout(config.drop_rate)

        # Stochastic-depth schedule: linearly scaled per block
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.depth)]
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=config.embed_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=config.qkv_bias,
                    drop=config.drop_rate,
                    attn_drop=config.attn_drop_rate,
                    drop_path=dpr[i],
                    layer_norm_eps=config.layer_norm_eps,
                )
                for i in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)

        self._init_weights()

    # ---------- initialization ----------
    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_layer)

    @staticmethod
    def _init_layer(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    # ---------- positional embedding interpolation ----------
    def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        """Interpolate the learned positional embedding to match input size.

        Required because DINO's multi-crop passes different-sized crops (global
        and local) through the same backbone.
        """
        n_patch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if n_patch == N and w == h:
            return self.pos_embed

        class_pos = self.pos_embed[:, 0]                 # (1, d)
        patch_pos = self.pos_embed[:, 1:]                # (1, N, d)
        dim = x.shape[-1]

        # grid size in number of patches along each side
        w0 = w // self.config.patch_size
        h0 = h // self.config.patch_size
        # side length of the original positional-embedding grid
        orig_side = int(math.sqrt(N))

        patch_pos = patch_pos.reshape(1, orig_side, orig_side, dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos, size=(h0, w0), mode="bicubic", align_corners=False
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat([class_pos.unsqueeze(1), patch_pos], dim=1)

    # ---------- forward ----------
    def prepare_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Patch-embed, prepend CLS, add positional embedding."""
        B, C, H, W = x.shape
        x = self.patch_embed(x)                           # (B, N, d)
        cls = self.cls_token.expand(B, -1, -1)            # (B, 1, d)
        x = torch.cat([cls, x], dim=1)                    # (B, N+1, d)
        x = x + self.interpolate_pos_encoding(x, W, H)
        return self.pos_drop(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning the (unnormalized) [CLS] token."""
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]                                    # [CLS] only

    def forward_all_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning (B, N+1, d) including CLS + patch tokens."""
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def get_last_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final-block attention weights (for visualization)."""
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i == len(self.blocks) - 1:
                _, attn = blk(x, return_attn=True)
                return attn
            x = blk(x)
        raise RuntimeError("unreachable")


# -----------------------------------------------------------------------------
# DINO projection head
# -----------------------------------------------------------------------------

class DINOHead(nn.Module):
    """Projection head used by DINO (Caron et al., 2021, Sec. 3).

    A 3-layer MLP (in_dim -> hidden -> hidden -> bottleneck) with GELU
    activations, followed by L2-normalization of the bottleneck features and a
    weight-normalized linear layer projecting to K prototype logits.

    Parameters
    ----------
    in_dim : int
        Backbone output dim, i.e. embed_dim.
    out_dim : int
        Prototype dimension K.
    hidden_dim : int
        Hidden dim of the MLP (default 2048, per DINO paper).
    bottleneck_dim : int
        Bottleneck dim before the final projection (default 256).
    norm_last_layer : bool
        If True, freeze the weight norm scale of the last linear layer,
        which stabilizes early training (recommended by the DINO authors).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 4096,          # K; smaller than the original 65536 for our mini setup
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        norm_last_layer: bool = True,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.apply(self._init_weights)

        # Weight-normalized linear layer for the final projection to prototypes.
        # Use the modern `parametrizations.weight_norm` API (PyTorch 2.0+).
        linear = nn.Linear(bottleneck_dim, out_dim, bias=False)
        self.last_layer = nn.utils.parametrizations.weight_norm(linear)
        # Under the parametrizations API, the weight-norm magnitude is
        # `.parametrizations.weight.original0` and the direction is `.original1`.
        # Fill the magnitude to 1 and optionally freeze it (per DINO's stabiliser).
        self.last_layer.parametrizations.weight.original0.data.fill_(1.0)
        if norm_last_layer:
            self.last_layer.parametrizations.weight.original0.requires_grad = False

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)                 # (B, bottleneck_dim)
        x = F.normalize(x, p=2, dim=-1) # L2-normalize on the hypersphere
        x = self.last_layer(x)          # (B, K)
        return x


# -----------------------------------------------------------------------------
# Combined model (backbone + head)
# -----------------------------------------------------------------------------

class DINOModel(nn.Module):
    """Backbone + DINO head, with a single call-path that handles a list of crops.

    Following the DINO reference implementation, when presented with a list of
    crops of different spatial sizes, we group crops of the same size, run them
    through the backbone in a single forward pass, and then concatenate all
    outputs before the head. This is a substantial throughput improvement over
    naively running each crop separately.
    """

    def __init__(self, backbone: VisionTransformer, head: DINOHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, crops: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass over a list of crops with possibly heterogeneous sizes.

        Parameters
        ----------
        crops : list of tensors, each (B, 3, H_i, W_i)

        Returns
        -------
        projections : (n_crops * B, K)
            Concatenated head outputs in the same crop order as the input list.
        """
        if not isinstance(crops, (list, tuple)):
            crops = [crops]

        # Group consecutive crops of identical spatial size for a batched forward
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([c.shape[-1] for c in crops]), return_counts=True
            )[1],
            dim=0,
        )

        start_idx, output = 0, torch.empty(0, device=crops[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(crops[start_idx:end_idx], dim=0))
            output = torch.cat((output, _out))
            start_idx = end_idx

        return self.head(output)


# -----------------------------------------------------------------------------
# Convenience constructors for common configs
# -----------------------------------------------------------------------------

def vit_tiny(image_size: int = 96, patch_size: int = 8, **kwargs) -> VisionTransformer:
    """ViT-Tiny: 5.5M params. A good starting point for CIFAR-style mini-training."""
    cfg = ViTConfig(
        image_size=image_size, patch_size=patch_size,
        embed_dim=192, depth=12, num_heads=3,
        **kwargs,
    )
    return VisionTransformer(cfg)


def vit_small(image_size: int = 96, patch_size: int = 8, **kwargs) -> VisionTransformer:
    """ViT-Small: 22M params."""
    cfg = ViTConfig(
        image_size=image_size, patch_size=patch_size,
        embed_dim=384, depth=12, num_heads=6,
        **kwargs,
    )
    return VisionTransformer(cfg)
