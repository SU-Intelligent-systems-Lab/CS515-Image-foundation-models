#!/usr/bin/env python
# Copyright (c) 2025. Released under the DINOv3 License Agreement.
# See NOTICE.md for details.
"""
segment_image.py
================

Run semantic segmentation on a single user-supplied image using either
the linear head or the Mask2Former head from the official DINOv3
repository (`facebookresearch/dinov3`).

This script does not implement segmentation; it only orchestrates the
pieces that already exist in Meta's `dinov3` Python package:

    1. Build a DINOv3 backbone via `dinov3.hub.backbones`.
    2. Build the segmentation decoder (linear or m2f) via
       `dinov3.eval.segmentation.models.build_segmentation_decoder`.
    3. Load the decoder checkpoint (with `strict=False`, the same way
       Meta's `dinov3.hub.segmentors._make_dinov3_m2f_segmentor` does it).
    4. Run sliding-window inference via
       `dinov3.eval.segmentation.inference.make_inference`,
       with `crop_size`/`stride` matching Meta's eval configs:
           - linear : crop=512, stride=341  (config-ade20k-linear-training.yaml)
           - m2f    : crop=896, stride=596  (config-ade20k-m2f-inference.yaml)
    5. Convert the per-pixel class logits to a colour overlay using the
       ADE20K palette and save it.

This script is the ONLY new code in this repository; everything else is
Meta's. See NOTICE.md and README.md.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Mean/std are the LVD-1689M defaults that DINOv3 expects for web-pretrained
# backbones (see Meta's README "Image transforms" section, and
# `dinov3/data/transforms.py`).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--head", choices=["linear", "m2f"], required=True,
        help="Which decoder head to use.",
    )
    p.add_argument(
        "--backbone", required=True,
        help=(
            "DINOv3 backbone hub identifier, e.g. 'dinov3_vits16', "
            "'dinov3_vitb16', 'dinov3_vitl16', 'dinov3_vit7b16'. "
            "Must match the function name in dinov3.hub.backbones."
        ),
    )
    p.add_argument(
        "--backbone-weights", required=True, type=Path,
        help="Path to the .pth file with pretrained backbone weights.",
    )
    p.add_argument(
        "--decoder-weights", required=True, type=Path,
        help=(
            "Path to the trained decoder-head .pth file. For 'linear', "
            "this is the file produced by Meta's training script. For "
            "'m2f', this is the released ADE20K head from Meta."
        ),
    )
    p.add_argument(
        "--image", required=True, type=Path,
        help="Path to the input image (any format readable by PIL).",
    )
    p.add_argument(
        "--output", required=True, type=Path,
        help="Path to write the colourised segmentation output.",
    )
    p.add_argument(
        "--num-classes", type=int, default=150,
        help="Number of segmentation classes. ADE20K = 150.",
    )
    p.add_argument(
        "--resize-short-side", type=int, default=None,
        help=(
            "If set, resize the image so its short side equals this value "
            "while preserving aspect ratio. Useful for very large inputs. "
            "Recommended: 512 for the linear head, 896 for M2F (matching "
            "Meta's eval-time img_size in the respective configs)."
        ),
    )
    p.add_argument(
        "--crop-size", type=int, default=None,
        help=(
            "Sliding-window crop size. Defaults to 512 (linear) or 896 "
            "(m2f) per Meta's eval configs."
        ),
    )
    p.add_argument(
        "--stride", type=int, default=None,
        help="Sliding-window stride. Defaults to 341 (linear) or 596 (m2f).",
    )
    p.add_argument(
        "--alpha", type=float, default=0.55,
        help="Overlay opacity for the segmentation map (0..1).",
    )
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--cmap", choices=["ade20k", "voc", "random"], default="ade20k",
        help="Colour palette to use for class IDs.",
    )
    return p.parse_args()


# ----------------------------------------------------------------------------
# Palette utilities
# ----------------------------------------------------------------------------

def _ade20k_palette(num_classes: int = 150) -> np.ndarray:
    """Deterministic palette for ADE20K's 150 classes.

    We do not embed Meta's authoritative palette file (some palettes have
    licensing concerns of their own); instead we generate a perceptually
    spread-out palette via HSV with a fixed seed. The result is
    deterministic across runs and machines, which is all we need to compare
    segmentation outputs from the same script.
    """
    rng = np.random.default_rng(seed=0xADE20)
    # Use HSV space and convert to RGB so colours are well-separated.
    hsv = np.stack([
        np.linspace(0.0, 1.0, num_classes, endpoint=False),
        rng.uniform(0.55, 1.0, num_classes),
        rng.uniform(0.55, 1.0, num_classes),
    ], axis=1)
    rgb = _hsv_to_rgb(hsv)
    palette = (rgb * 255).astype(np.uint8)
    # Class 0 stays black to highlight ignore / background.
    palette[0] = 0
    return palette


def _voc_palette(num_classes: int = 21) -> np.ndarray:
    """Standard PASCAL VOC palette (deterministic bitwise construction)."""
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(num_classes):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        palette[i] = (r, g, b)
    return palette


def _random_palette(num_classes: int) -> np.ndarray:
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 256, size=(num_classes, 3), dtype=np.uint8)


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Vectorised HSV -> RGB. Inputs in [0, 1]."""
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    i = np.floor(h * 6).astype(int)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i = i % 6
    out = np.zeros_like(hsv)
    masks = [i == k for k in range(6)]
    rgbs = [
        (v, t, p), (q, v, p), (p, v, t),
        (p, q, v), (t, p, v), (v, p, q),
    ]
    for m, (r, g, b) in zip(masks, rgbs):
        out[m, 0] = r[m]; out[m, 1] = g[m]; out[m, 2] = b[m]
    return out


def get_palette(name: str, num_classes: int) -> np.ndarray:
    if name == "ade20k":
        return _ade20k_palette(num_classes)
    if name == "voc":
        return _voc_palette(num_classes)
    return _random_palette(num_classes)


# ----------------------------------------------------------------------------
# Image I/O
# ----------------------------------------------------------------------------

def load_image_tensor(
    path: Path, resize_short_side: int | None, device: str
) -> tuple[torch.Tensor, tuple[int, int], np.ndarray]:
    """Load an image and return (CHW float tensor, (H,W) of the tensor, RGB uint8 array of the original)."""
    img = Image.open(path).convert("RGB")
    original_rgb = np.asarray(img)

    if resize_short_side is not None:
        w0, h0 = img.size
        scale = resize_short_side / float(min(w0, h0))
        new_size = (int(round(w0 * scale)), int(round(h0 * scale)))
        img = img.resize(new_size, Image.BICUBIC)

    arr = np.asarray(img).astype(np.float32) / 255.0     # H,W,C
    arr = (arr - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()
    tensor = tensor.unsqueeze(0).to(device)              # 1,C,H,W
    H, W = tensor.shape[-2], tensor.shape[-1]
    return tensor, (H, W), original_rgb


def colourise_prediction(
    pred: np.ndarray, palette: np.ndarray, original_rgb: np.ndarray, alpha: float
) -> np.ndarray:
    """Map class IDs to colours and blend with the original image."""
    coloured = palette[pred]                              # H,W,3 uint8
    if coloured.shape[:2] != original_rgb.shape[:2]:
        # Resize the colour map to the original input resolution.
        from PIL import Image as _Image
        coloured = np.asarray(
            _Image.fromarray(coloured).resize(
                (original_rgb.shape[1], original_rgb.shape[0]),
                _Image.NEAREST,
            )
        )
    blend = (alpha * coloured + (1 - alpha) * original_rgb).clip(0, 255).astype(np.uint8)
    return blend


# ----------------------------------------------------------------------------
# Model construction (delegates to Meta's package)
# ----------------------------------------------------------------------------

def build_backbone(name: str, weights_path: Path):
    """Call into dinov3.hub.backbones.<name>(weights=<path>)."""
    try:
        backbones_mod = importlib.import_module("dinov3.hub.backbones")
    except ImportError as e:
        sys.exit(
            "Could not import dinov3. Have you cloned and installed Meta's "
            "DINOv3 repo? See README.md, sections 2 and 3.\n"
            f"(Underlying error: {e})"
        )
    factory = getattr(backbones_mod, name, None)
    if factory is None:
        sys.exit(
            f"No backbone factory called '{name}' in dinov3.hub.backbones. "
            f"Try one of: dinov3_vits16, dinov3_vitb16, dinov3_vitl16, "
            f"dinov3_vit7b16."
        )
    # `weights` may be a path or a Weights enum value; passing a path triggers
    # convert_path_or_url_to_url() inside Meta's hub code.
    return factory(pretrained=True, weights=str(weights_path))


def build_decoder(backbone, head: str, num_classes: int):
    """Build the decoder via Meta's build_segmentation_decoder().

    For the linear path, Meta uses BackboneLayersSet.LAST in the ADE20K
    linear training config; for m2f, FOUR_EVEN_INTERVALS. We mirror that.
    """
    seg_models = importlib.import_module("dinov3.eval.segmentation.models")
    BackboneLayersSet = seg_models.BackboneLayersSet
    layers = (
        BackboneLayersSet.LAST if head == "linear"
        else BackboneLayersSet.FOUR_EVEN_INTERVALS
    )
    decoder = seg_models.build_segmentation_decoder(
        backbone_model=backbone,
        backbone_out_layers=layers,
        decoder_type=head,
        num_classes=num_classes,
        autocast_dtype=torch.bfloat16 if head == "m2f" else torch.float32,
    )
    return decoder


def load_decoder_weights(decoder, ckpt_path: Path) -> None:
    """Load decoder-head weights, mirroring Meta's segmentor hub loader."""
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = decoder.load_state_dict(state, strict=False)
    # Meta's loader (dinov3/hub/segmentors.py:_make_dinov3_m2f_segmentor)
    # asserts that any missing keys must be backbone keys (we keep backbone
    # frozen and load it separately). We replicate that check.
    non_backbone_missing = [k for k in missing if "backbone" not in k]
    if non_backbone_missing:
        print(
            f"Warning: {len(non_backbone_missing)} non-backbone keys missing "
            f"from decoder checkpoint:"
        )
        for k in non_backbone_missing[:10]:
            print(f"  - {k}")
    if unexpected:
        print(f"Warning: {len(unexpected)} unexpected keys in decoder checkpoint:")
        for k in unexpected[:10]:
            print(f"  - {k}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Defaults for crop/stride mirror Meta's eval configs.
    if args.crop_size is None:
        args.crop_size = 512 if args.head == "linear" else 896
    if args.stride is None:
        args.stride = 341 if args.head == "linear" else 596
    if args.resize_short_side is None:
        args.resize_short_side = args.crop_size

    # Load the user image.
    img_tensor, (H, W), original_rgb = load_image_tensor(
        args.image, args.resize_short_side, args.device
    )
    print(f"Input image resized to {H}x{W} on {args.device}.")

    # Build backbone + decoder.
    print(f"Building backbone '{args.backbone}'...")
    backbone = build_backbone(args.backbone, args.backbone_weights)
    backbone.eval()

    print(f"Building '{args.head}' decoder for {args.num_classes} classes...")
    decoder = build_decoder(backbone, args.head, args.num_classes)
    load_decoder_weights(decoder, args.decoder_weights)
    decoder.to(args.device).eval()

    # Sliding-window inference via Meta's `make_inference`.
    inference_mod = importlib.import_module("dinov3.eval.segmentation.inference")
    make_inference = inference_mod.make_inference

    print(
        f"Running sliding-window inference: crop={args.crop_size}, "
        f"stride={args.stride}..."
    )
    with torch.inference_mode():
        # Output activation is softmax for multiclass segmentation.
        pred = make_inference(
            x=img_tensor,
            segmentation_model=decoder,
            inference_mode="slide",
            decoder_head_type=args.head,
            rescale_to=(H, W),
            n_output_channels=args.num_classes,
            crop_size=(args.crop_size, args.crop_size),
            stride=(args.stride, args.stride),
            output_activation=lambda x: F.softmax(x, dim=1),
        )

    # Reduce to per-pixel class IDs.
    class_map = pred.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)

    # Colourise and save.
    palette = get_palette(args.cmap, args.num_classes)
    overlay = colourise_prediction(class_map, palette, original_rgb, args.alpha)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(args.output)
    print(f"Saved colour overlay to {args.output}")

    # Also save the raw class map for downstream metrics.
    raw_path = args.output.with_suffix(".classes.png")
    Image.fromarray(class_map.astype(np.uint16)).save(raw_path)
    print(f"Saved raw class map to {raw_path}")


if __name__ == "__main__":
    main()
