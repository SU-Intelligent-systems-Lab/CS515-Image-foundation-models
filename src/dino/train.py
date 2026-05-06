"""
Training utilities for the mini DINO reimplementation.

Exports
-------
train_one_epoch
    Runs one full epoch through the training loader, updating the student
    parameters, the EMA teacher, and the DINO loss's centering buffer.

build_optimizer
    Helper that constructs an AdamW optimizer with appropriate parameter
    groups (no weight decay on biases and LayerNorm weights, following
    the DINO reference implementation).
"""

from __future__ import annotations

import math
import sys
import time
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .loss import DINOLoss
from .teacher_student import cancel_gradients_last_layer, update_teacher_ema


# -----------------------------------------------------------------------------
# Optimizer builder
# -----------------------------------------------------------------------------

def _get_params_groups(model: nn.Module) -> List[Dict]:
    """Split parameters into two groups: those that receive weight decay and
    those that do not (biases and all 1-D parameters such as LayerNorm weights).
    """
    regularized, not_regularized = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Biases and normalization weights: no weight decay
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [
        {"params": regularized},
        {"params": not_regularized, "weight_decay": 0.0},
    ]


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    optimizer: str = "adamw",
) -> torch.optim.Optimizer:
    """Construct an optimizer for the student, with correct no-WD treatment of
    biases and norms. ``lr`` and ``weight_decay`` are placeholders that will
    be overwritten by the scheduler each step.
    """
    param_groups = _get_params_groups(model)
    optimizer = optimizer.lower()
    if optimizer == "adamw":
        return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    if optimizer == "sgd":
        return torch.optim.SGD(
            param_groups, lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    raise ValueError(f"Unknown optimizer: {optimizer}")


# -----------------------------------------------------------------------------
# Gradient clipping helper
# -----------------------------------------------------------------------------

def clip_gradients(model: nn.Module, max_norm: float) -> float:
    """Clip gradients and return the total (pre-clipping) grad-norm for logging."""
    total_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.grad is not None],
        max_norm=max_norm,
    )
    return float(total_norm)


# -----------------------------------------------------------------------------
# Training step and epoch
# -----------------------------------------------------------------------------

def train_one_epoch(
    student: nn.Module,
    teacher: nn.Module,
    loss_fn: DINOLoss,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_schedule: np.ndarray,
    wd_schedule: np.ndarray,
    momentum_schedule_arr: np.ndarray,
    teacher_temp_schedule: np.ndarray,
    epoch: int,
    total_epochs: int,
    freeze_last_layer_epochs: int,
    clip_grad: float,
    device: torch.device,
    global_step_start: int,
    log_every: int = 20,
    use_amp: bool = True,
) -> Tuple[int, Dict[str, float]]:
    """Train DINO for one epoch.

    Returns
    -------
    global_step_end : int
        The global-step counter after processing this epoch (= number of
        optimizer steps executed across all epochs so far).
    metrics : dict
        Mean loss and other scalars for logging.
    """
    student.train()
    teacher.eval()  # teacher never accumulates BatchNorm stats / dropout

    scaler = torch.amp.GradScaler(enabled=use_amp)

    running_loss, running_grad_norm, n_batches = 0.0, 0.0, 0
    global_step = global_step_start

    pbar = tqdm(
        data_loader,
        desc=f"Epoch {epoch + 1}/{total_epochs}",
        leave=False,
        file=sys.stdout,
    )
    for batch in pbar:
        # Dataloader returns (list of crops, label) when using DINOMultiCropTransform
        # bundled through a dataset whose __getitem__ calls it. We ignore the label.
        crops, _ = batch

        # Move all crops to device
        crops = [c.to(device, non_blocking=True) for c in crops]

        # --- Schedules (per-step) ---
        for i, pg in enumerate(optimizer.param_groups):
            pg["lr"] = float(lr_schedule[global_step])
            if i == 0:                                   # regularized group
                pg["weight_decay"] = float(wd_schedule[global_step])
        loss_fn.teacher_temp = float(teacher_temp_schedule[global_step])

        # --- Forward ---
        with torch.amp.autocast(
            device_type=device.type, enabled=use_amp, dtype=torch.float16
        ):
            # Teacher sees only global crops (first n_global of the list)
            with torch.no_grad():
                teacher_out = teacher(crops[: loss_fn.n_global_crops])
            # Student sees all crops
            student_out = student(crops)
            loss = loss_fn(student_out, teacher_out)

        if not torch.isfinite(loss):
            print(f"[warn] non-finite loss at step {global_step}: {loss}. Skipping.")
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            continue

        # --- Backward / step ---
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        if clip_grad > 0:
            scaler.unscale_(optimizer)
            grad_norm = clip_gradients(student, clip_grad)
        else:
            grad_norm = 0.0

        cancel_gradients_last_layer(student, epoch, freeze_last_layer_epochs)
        scaler.step(optimizer)
        scaler.update()

        # --- EMA teacher update ---
        with torch.no_grad():
            m = float(momentum_schedule_arr[global_step])
            update_teacher_ema(student, teacher, momentum=m)

        # --- Logging ---
        running_loss += float(loss.detach())
        running_grad_norm += grad_norm
        n_batches += 1
        if global_step % log_every == 0:
            pbar.set_postfix(
                loss=f"{float(loss.detach()):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                tT=f"{loss_fn.teacher_temp:.3f}",
                m=f"{m:.4f}",
            )

        global_step += 1

    metrics = {
        "train/loss": running_loss / max(n_batches, 1),
        "train/grad_norm": running_grad_norm / max(n_batches, 1),
        "train/lr": float(optimizer.param_groups[0]["lr"]),
        "train/teacher_temp": loss_fn.teacher_temp,
    }
    return global_step, metrics
