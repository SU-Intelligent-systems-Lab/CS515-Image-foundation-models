"""
Run the feature-analysis pipeline as a non-interactive script.

This is the script-form equivalent of ``notebooks/01_dino_feature_exploration``.
It produces the figures and tables listed in the config under the ``output:``
section. Useful for regenerating report figures from the command line.

Usage
-----
    python scripts/run_feature_analysis.py --config configs/feature_analysis.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "dinovpr", _REPO_ROOT / "src" / "__init__.py",
    submodule_search_locations=[str(_REPO_ROOT / "src")],
)
_m = importlib.util.module_from_spec(_spec); sys.modules["dinovpr"] = _m
_spec.loader.exec_module(_m)

import numpy as np
import torch
from torch.utils.data import DataLoader

from dinovpr.data.datasets import build_cifar100
from dinovpr.data.transforms import build_eval_transform
from dinovpr.utils.io import get_device, load_config, set_seed
from dinovpr.utils.feature_analysis import extract_features, knn_classify


def load_hf_dinov2(hf_id: str, device: torch.device):
    """Load a HuggingFace DINOv2 model and return a (model, cls_extractor) pair."""
    try:
        from transformers import AutoModel, AutoImageProcessor
    except ImportError as exc:  # pragma: no cover - guard for optional dep
        raise RuntimeError(
            "`transformers` is required for feature_analysis.py; "
            "install with `pip install transformers`."
        ) from exc

    model = AutoModel.from_pretrained(hf_id).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(hf_id)

    def extractor(x: torch.Tensor) -> torch.Tensor:
        # HuggingFace DINOv2 exposes `.pooler_output` (CLS after layernorm).
        out = model(pixel_values=x)
        return out.pooler_output
    return model, processor, extractor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])
    device = get_device()
    print(f"[cfg] {args.config}  [device] {device}")

    # ---- Build model ----
    mcfg = cfg["model"]
    if mcfg["source"] != "huggingface":
        raise NotImplementedError("Only source='huggingface' is wired up for now.")
    print(f"[model] loading {mcfg['hf_id']} ...")
    model, processor, extract_cls = load_hf_dinov2(mcfg["hf_id"], device)
    print(f"[model] loaded {mcfg['hf_id']}")

    # ---- Build eval data ----
    image_size = int(mcfg["image_size"])
    eval_tf = build_eval_transform(image_size=image_size)

    ds_train = build_cifar100(
        root=cfg["data"]["root"], train=True, eval_transform=eval_tf,
        download=cfg["data"].get("download", True),
    )
    ds_val = build_cifar100(
        root=cfg["data"]["root"], train=False, eval_transform=eval_tf,
        download=False,
    )
    loader_kw = dict(batch_size=cfg["data"]["batch_size"],
                     num_workers=cfg["data"]["num_workers"], pin_memory=True)
    loader_train = DataLoader(ds_train, shuffle=False, **loader_kw)
    loader_val = DataLoader(ds_val, shuffle=False, **loader_kw)

    # ---- kNN probe ----
    if cfg["quantitative"]["run_knn"]:
        print("[knn] extracting features ...")
        train_feats, train_labels = extract_features(extract_cls, loader_train, device,
                                                     desc="kNN train")
        val_feats, val_labels = extract_features(extract_cls, loader_val, device,
                                                 desc="kNN val")
        res = knn_classify(
            train_feats, train_labels, val_feats, val_labels,
            num_classes=100,
            k=cfg["quantitative"]["knn_k"],
            temperature=cfg["quantitative"]["knn_temperature"],
        )
        print(f"[knn] top-1={res['top1']:.2f}%  top-5={res['top5']:.2f}%")

        # Dump to CSV
        table_path = Path(cfg["output"]["table_csv"])
        table_path.parent.mkdir(parents=True, exist_ok=True)
        with open(table_path, "w", encoding="utf-8") as f:
            f.write("metric,value\n")
            for k, v in res.items():
                f.write(f"{k},{v}\n")
        print(f"[knn] summary saved -> {table_path}")


if __name__ == "__main__":
    main()
