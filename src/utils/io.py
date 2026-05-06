"""
Small IO / bookkeeping helpers.

Contents
--------
load_config
    Read a YAML config into a nested dict, optionally resolving ${env} refs.
save_checkpoint / load_checkpoint
    Robust checkpoint helpers that handle student, teacher, optimizer, scheduler
    state, RNG state, and the DINOLoss centering buffer.
set_seed
    Seed all the usual sources of nondeterminism.
get_device
    Return the CUDA device if available, else CPU, honoring CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


# -----------------------------------------------------------------------------
# Config loading
# -----------------------------------------------------------------------------

def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    """Load a YAML config file into a dictionary.

    Supports simple ``${env:VAR}`` substitution for string values
    (e.g. ``root: ${env:DATASETS_DIR}/cifar100``).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return _resolve_env(cfg)


def _resolve_env(obj: Any) -> Any:
    """Recursively replace ${env:VAR} patterns in string values."""
    if isinstance(obj, dict):
        return {k: _resolve_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env(v) for v in obj]
    if isinstance(obj, str) and "${env:" in obj:
        out = obj
        while "${env:" in out:
            start = out.index("${env:")
            end = out.index("}", start)
            var = out[start + 6 : end]
            out = out[:start] + os.environ.get(var, "") + out[end + 1 :]
        return out
    return obj


# -----------------------------------------------------------------------------
# Checkpointing
# -----------------------------------------------------------------------------

def save_checkpoint(
    path: str | os.PathLike,
    *,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    dino_loss_center: Optional[torch.Tensor] = None,
    epoch: Optional[int] = None,
    global_step: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a training checkpoint atomically (write to .tmp, then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "student": student.state_dict(),
        "teacher": teacher.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if dino_loss_center is not None:
        payload["dino_loss_center"] = dino_loss_center.detach().cpu()
    if extra is not None:
        payload["extra"] = extra

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(
    path: str | os.PathLike,
    *,
    student: Optional[torch.nn.Module] = None,
    teacher: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """Load a checkpoint and optionally restore modules in place."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if student is not None and "student" in payload:
        student.load_state_dict(payload["student"], strict=strict)
    if teacher is not None and "teacher" in payload:
        teacher.load_state_dict(payload["teacher"], strict=strict)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + all CUDA devices).

    ``deterministic=True`` is heavy: it disables cuDNN's nondeterministic
    algorithms and is typically only used when diagnosing reproducibility bugs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device(prefer: str = "cuda") -> torch.device:
    """Return a torch.device, falling back gracefully to CPU."""
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and getattr(torch.backends, "mps", None) is not None:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")
