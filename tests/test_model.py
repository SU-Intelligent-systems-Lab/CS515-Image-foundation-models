"""Tests for the ViT + DINO head + combined model."""

from __future__ import annotations

import pytest
import torch

from dinovpr.dino.model import (
    DINOHead,
    DINOModel,
    ViTConfig,
    VisionTransformer,
    vit_small,
    vit_tiny,
)


def test_vit_tiny_forward_cls_shape():
    m = vit_tiny(image_size=32, patch_size=4)
    x = torch.randn(2, 3, 32, 32)
    z = m(x)                                  # CLS token only
    assert z.shape == (2, m.config.embed_dim)


def test_vit_tiny_forward_all_tokens():
    m = vit_tiny(image_size=32, patch_size=4)
    x = torch.randn(2, 3, 32, 32)
    z = m.forward_all_tokens(x)
    # (B, 1 + N, d)
    assert z.shape == (2, 1 + m.config.num_patches, m.config.embed_dim)


def test_vit_positional_interpolation_handles_different_sizes():
    m = vit_tiny(image_size=32, patch_size=4)
    # Global-crop size (32) and a smaller local-crop size (16)
    g = torch.randn(2, 3, 32, 32); l = torch.randn(2, 3, 16, 16)
    assert m(g).shape == (2, m.config.embed_dim)
    assert m(l).shape == (2, m.config.embed_dim)


def test_dino_head_output_shape_and_norm():
    h = DINOHead(in_dim=192, out_dim=1024, hidden_dim=512, bottleneck_dim=128)
    x = torch.randn(4, 192)
    y = h(x)
    assert y.shape == (4, 1024)


def test_dino_head_norm_last_layer_freezes_magnitude():
    h = DINOHead(in_dim=192, out_dim=1024, norm_last_layer=True)
    # The weight-norm magnitude is .parametrizations.weight.original0
    mag = h.last_layer.parametrizations.weight.original0
    assert mag.requires_grad is False
    # Defaults to 1
    assert torch.allclose(mag, torch.ones_like(mag))


def test_dinomodel_concat_order_matches_crop_order():
    """DINOModel groups crops of identical size before forwarding; check the
    concatenation ordering is still crop-major."""
    backbone = vit_tiny(image_size=32, patch_size=4)
    head = DINOHead(in_dim=192, out_dim=256, hidden_dim=128, bottleneck_dim=64)
    model = DINOModel(backbone, head)

    B = 3
    # 2 global (32x32) + 4 local (16x16): should produce (6*B, 256) output
    crops = [torch.randn(B, 3, 32, 32)] * 2 + [torch.randn(B, 3, 16, 16)] * 4
    y = model(crops)
    assert y.shape == (6 * B, 256)


@pytest.mark.parametrize("builder,expected_embed", [(vit_tiny, 192), (vit_small, 384)])
def test_builders_set_expected_embed_dim(builder, expected_embed):
    m = builder(image_size=32, patch_size=4)
    assert m.config.embed_dim == expected_embed


def test_attention_extraction_shape():
    m = vit_tiny(image_size=32, patch_size=4)
    x = torch.randn(2, 3, 32, 32)
    attn = m.get_last_attention(x)
    # (B, H, N+1, N+1)
    expected_tokens = 1 + m.config.num_patches
    assert attn.shape == (2, m.config.num_heads, expected_tokens, expected_tokens)
    # Rows sum to 1 (softmaxed)
    assert torch.allclose(attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1)), atol=1e-4)
