"""
Mini DINO training entrypoint.

Usage
-----
    python scripts/run_mini_dino.py --config configs/mini_dino_cifar100.yaml

or, when installed as the package console script:

    dinovpr-train-mini --config configs/mini_dino_cifar100.yaml

This script is intentionally thin: it parses the YAML config, builds the
student, teacher, loss, data loaders, and schedules; then hands off to
``dinovpr.dino.train.train_one_epoch`` for each epoch. Every 5 epochs (by
default) it runs a kNN eval on the CIFAR-100 test set and logs the result.

All logs, checkpoints, and figures land in the output directory specified in
the config.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

# Make the `src/` package importable as `dinovpr` regardless of where the
# script is invoked from.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

# Register the `src` folder as the `dinovpr` package.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "dinovpr", _REPO_ROOT / "src" / "__init__.py",
    submodule_search_locations=[str(_REPO_ROOT / "src")],
)
_dinovpr = importlib.util.module_from_spec(_spec)
sys.modules["dinovpr"] = _dinovpr
_spec.loader.exec_module(_dinovpr)

from dinovpr.dino.model import DINOHead, DINOModel, vit_small, vit_tiny
from dinovpr.dino.loss import DINOLoss
from dinovpr.dino.augmentation import DINOMultiCropTransform
from dinovpr.dino.teacher_student import (
    cosine_schedule,
    deactivate_requires_grad,
    momentum_schedule,
)
from dinovpr.dino.train import build_optimizer, train_one_epoch
from dinovpr.data.datasets import (
    build_cifar100,
    build_eval_transform,
    multicrop_collate_fn,
)
from dinovpr.utils.io import get_device, load_config, save_checkpoint, set_seed
from dinovpr.utils.feature_analysis import extract_features, knn_classify
from dinovpr.utils.visualization import plot_loss_curves


# -----------------------------------------------------------------------------
BACKBONE_BUILDERS = {"vit_tiny": vit_tiny, "vit_small": vit_small}


# -----------------------------------------------------------------------------

def build_student_teacher(cfg: Dict, device: torch.device):
    mcfg = cfg["model"]
    if mcfg["backbone"] not in BACKBONE_BUILDERS:
        raise ValueError(f"Unknown backbone: {mcfg['backbone']}")
    builder = BACKBONE_BUILDERS[mcfg["backbone"]]

    # Both student and teacher use the same architecture
    image_size = cfg["data"]["augmentation"]["global_size"]
    patch_size = mcfg["patch_size"]

    student_backbone = builder(
        image_size=image_size, patch_size=patch_size,
        drop_path_rate=mcfg.get("drop_path_rate", 0.0),
    )
    teacher_backbone = builder(
        image_size=image_size, patch_size=patch_size, drop_path_rate=0.0,
    )

    head_cfg = mcfg["head"]
    embed_dim = student_backbone.config.embed_dim
    student_head = DINOHead(
        in_dim=embed_dim,
        out_dim=head_cfg["out_dim"],
        hidden_dim=head_cfg["hidden_dim"],
        bottleneck_dim=head_cfg["bottleneck_dim"],
        norm_last_layer=head_cfg["norm_last_layer"],
    )
    teacher_head = DINOHead(
        in_dim=embed_dim,
        out_dim=head_cfg["out_dim"],
        hidden_dim=head_cfg["hidden_dim"],
        bottleneck_dim=head_cfg["bottleneck_dim"],
        norm_last_layer=head_cfg["norm_last_layer"],
    )

    student = DINOModel(student_backbone, student_head).to(device)
    teacher = DINOModel(teacher_backbone, teacher_head).to(device)

    # Teacher is initialized to the student's weights and then frozen.
    teacher.load_state_dict(student.state_dict())
    deactivate_requires_grad(teacher)

    return student, teacher


def build_train_loader(cfg: Dict) -> DataLoader:
    acfg = cfg["data"]["augmentation"]
    mc = DINOMultiCropTransform(
        global_size=acfg["global_size"],
        local_size=acfg["local_size"],
        n_global_crops=acfg["n_global_crops"],
        n_local_crops=acfg["n_local_crops"],
        global_scale=tuple(acfg["global_scale"]),
        local_scale=tuple(acfg["local_scale"]),
    )
    ds = build_cifar100(
        root=cfg["data"]["root"], train=True,
        multi_crop_transform=mc, download=cfg["data"].get("download", True),
    )
    return DataLoader(
        ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"].get("pin_memory", True),
        drop_last=True,
        collate_fn=multicrop_collate_fn,
        persistent_workers=cfg["data"]["num_workers"] > 0,
    )


def build_eval_loaders(cfg: Dict):
    """Build (train, val) loaders for kNN evaluation of the teacher backbone."""
    # Eval uses the global-crop resolution so the positional-embedding interpolation
    # is a no-op and features are computed at training resolution.
    image_size = cfg["data"]["augmentation"]["global_size"]
    eval_tf = build_eval_transform(image_size=image_size, resize_size=image_size)
    train_eval = build_cifar100(
        root=cfg["data"]["root"], train=True, eval_transform=eval_tf,
        download=False,
    )
    val_eval = build_cifar100(
        root=cfg["data"]["root"], train=False, eval_transform=eval_tf,
        download=False,
    )
    loader_kwargs = dict(
        batch_size=256,
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    return (
        DataLoader(train_eval, **loader_kwargs),
        DataLoader(val_eval, **loader_kwargs),
    )


@torch.no_grad()
def knn_eval(
    teacher: DINOModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_classes: int = 100,
    k: int = 20,
    temperature: float = 0.07,
) -> Dict:
    """kNN eval on the teacher backbone's CLS features (no head)."""
    # Use the backbone only (strip the head) for downstream probing.
    def extractor(x: torch.Tensor) -> torch.Tensor:
        return teacher.backbone(x)

    train_f, train_y = extract_features(extractor, train_loader, device, desc="kNN eval (train)")
    val_f, val_y = extract_features(extractor, val_loader, device, desc="kNN eval (val)")
    return knn_classify(train_f, train_y, val_f, val_y, num_classes=num_classes, k=k, temperature=temperature)


# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Mini DINO training entrypoint.")
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    ap.add_argument("--resume", type=str, default=None, help="Optional checkpoint to resume from.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])
    device = get_device()
    print(f"[config] {args.config}")
    print(f"[device] {device}")

    # ---- Build dataset / loader first, so we know steps_per_epoch ----
    train_loader = build_train_loader(cfg)
    steps_per_epoch = len(train_loader)
    epochs = cfg["training"]["epochs"]
    total_steps = steps_per_epoch * epochs
    print(f"[data] steps_per_epoch={steps_per_epoch}, total_steps={total_steps}")

    # ---- Build student / teacher ----
    student, teacher = build_student_teacher(cfg, device)
    num_params = sum(p.numel() for p in student.parameters()) / 1e6
    print(f"[model] student params: {num_params:.2f}M ({cfg['model']['backbone']})")

    # ---- Build loss ----
    loss_fn = DINOLoss(
        out_dim=cfg["model"]["head"]["out_dim"],
        n_global_crops=cfg["data"]["augmentation"]["n_global_crops"],
        n_local_crops=cfg["data"]["augmentation"]["n_local_crops"],
        student_temp=cfg["loss"]["student_temp"],
        teacher_temp=cfg["loss"]["teacher_temp_start"],
        center_momentum=cfg["loss"]["center_momentum"],
    ).to(device)

    # ---- Build optimizer + schedules ----
    optimizer = build_optimizer(
        student,
        lr=cfg["optim"]["base_lr"],                 # placeholder; scheduler overrides per-step
        weight_decay=cfg["optim"]["weight_decay"],
        optimizer=cfg["optim"]["optimizer"],
    )
    lr_schedule = cosine_schedule(
        start_value=cfg["optim"]["base_lr"],
        end_value=cfg["optim"]["min_lr"],
        total_steps=total_steps,
        warmup_steps=cfg["optim"]["warmup_epochs"] * steps_per_epoch,
    )
    wd_schedule = cosine_schedule(
        start_value=cfg["optim"]["weight_decay"],
        end_value=cfg["optim"]["weight_decay_end"],
        total_steps=total_steps,
    )
    momentum_schedule_arr = momentum_schedule(
        base_momentum=cfg["ema"]["base_momentum"],
        final_momentum=cfg["ema"]["final_momentum"],
        total_steps=total_steps,
    )
    teacher_temp_schedule = cosine_schedule(
        start_value=cfg["loss"]["teacher_temp_start"],
        end_value=cfg["loss"]["teacher_temp_end"],
        total_steps=total_steps,
        warmup_steps=cfg["loss"]["teacher_temp_warmup_epochs"] * steps_per_epoch,
        warmup_start_value=cfg["loss"]["teacher_temp_start"],
    )

    # ---- Resume ----
    start_epoch, global_step = 0, 0
    if args.resume:
        from dinovpr.utils.io import load_checkpoint
        payload = load_checkpoint(args.resume, student=student, teacher=teacher, optimizer=optimizer)
        if "dino_loss_center" in payload:
            loss_fn.center = payload["dino_loss_center"].to(device)
        start_epoch = (payload.get("epoch") or 0) + 1
        global_step = payload.get("global_step") or 0
        print(f"[resume] from {args.resume} at epoch {start_epoch}, step {global_step}")

    # ---- Output directory ----
    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Eval loaders (built once, reused) ----
    eval_train_loader, eval_val_loader = build_eval_loaders(cfg)

    # ---- Main loop ----
    history: Dict[str, List[float]] = {
        "train/loss": [], "train/grad_norm": [], "train/lr": [],
        "train/teacher_temp": [], "eval/knn_top1": [],
    }
    t0 = time.time()

    for epoch in range(start_epoch, epochs):
        global_step, metrics = train_one_epoch(
            student=student,
            teacher=teacher,
            loss_fn=loss_fn,
            data_loader=train_loader,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            wd_schedule=wd_schedule,
            momentum_schedule_arr=momentum_schedule_arr,
            teacher_temp_schedule=teacher_temp_schedule,
            epoch=epoch,
            total_epochs=epochs,
            freeze_last_layer_epochs=cfg["optim"]["freeze_last_layer_epochs"],
            clip_grad=cfg["optim"]["clip_grad"],
            device=device,
            global_step_start=global_step,
            log_every=cfg["training"]["log_every"],
            use_amp=cfg["training"].get("use_amp", True),
        )
        for k in ("train/loss", "train/grad_norm", "train/lr", "train/teacher_temp"):
            history[k].append(metrics[k])

        # ---- Eval ----
        eval_every = cfg["training"].get("eval_every_epochs", 5)
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            ev = knn_eval(
                teacher, eval_train_loader, eval_val_loader, device=device,
                num_classes=100, k=cfg["eval"]["knn_k"],
                temperature=cfg["eval"]["knn_temperature"],
            )
            history["eval/knn_top1"].append(ev["top1"])
            print(f"[epoch {epoch + 1}] loss={metrics['train/loss']:.4f}  knn_top1={ev['top1']:.2f}%")
        else:
            history["eval/knn_top1"].append(float("nan"))
            print(f"[epoch {epoch + 1}] loss={metrics['train/loss']:.4f}")

        # ---- Checkpoint ----
        save_every = cfg["training"].get("save_every_epochs", 10)
        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            ckpt_path = output_dir / f"ckpt_epoch_{epoch + 1:03d}.pth"
            save_checkpoint(
                ckpt_path, student=student, teacher=teacher, optimizer=optimizer,
                dino_loss_center=loss_fn.center, epoch=epoch, global_step=global_step,
                extra={"config": cfg},
            )
            print(f"[ckpt ] saved -> {ckpt_path}")

    # ---- Save final artefacts ----
    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    plot_loss_curves(history, out_path=output_dir / "training_curves.png",
                     title=cfg["experiment"]["name"])

    total_min = (time.time() - t0) / 60
    print(f"[done ] training took {total_min:.1f} minutes. Artifacts in {output_dir}")


if __name__ == "__main__":
    main()
