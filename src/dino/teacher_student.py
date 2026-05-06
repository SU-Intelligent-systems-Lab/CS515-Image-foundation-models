"""
Teacher/student utilities for DINO.

Contents
--------
update_teacher_ema
    In-place EMA update of teacher parameters from student parameters:
    ``theta_t <- lambda * theta_t + (1 - lambda) * theta_s``.

cosine_schedule
    Generic cosine schedule between two values over a number of training steps.
    Used for learning-rate warm-up/decay and weight-decay schedules.

momentum_schedule
    Cosine schedule specialized to DINO's teacher momentum, which typically
    ramps from ``base_momentum`` (e.g. 0.996) to 1.0 over the whole run.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Teacher EMA update
# -----------------------------------------------------------------------------

@torch.no_grad()
def update_teacher_ema(
    student: nn.Module,
    teacher: nn.Module,
    momentum: float,
) -> None:
    """Update teacher parameters as an EMA of the student parameters.

    .. math::

        \\theta_t \\leftarrow \\lambda\\,\\theta_t + (1-\\lambda)\\,\\theta_s

    Operates in place on ``teacher`` and must be called after the student's
    optimizer step at each iteration. Both networks must have exactly matching
    parameter structure (which is the case in DINO, where they share the
    architecture).

    Parameters
    ----------
    student, teacher : nn.Module
        The student and teacher networks. ``teacher`` must have
        ``requires_grad=False`` everywhere.
    momentum : float
        Coefficient :math:`\\lambda \\in [0, 1)` (often called the teacher
        momentum). DINO uses a cosine schedule from 0.996 to 1.0.
    """
    for p_s, p_t in zip(student.parameters(), teacher.parameters()):
        p_t.data.mul_(momentum).add_(p_s.detach().data, alpha=1.0 - momentum)


# -----------------------------------------------------------------------------
# Schedules
# -----------------------------------------------------------------------------

def cosine_schedule(
    start_value: float,
    end_value: float,
    total_steps: int,
    warmup_steps: int = 0,
    warmup_start_value: float = 0.0,
) -> np.ndarray:
    """Return a 1D numpy array of length ``total_steps`` with values following
    a linear warm-up then a cosine decay from ``start_value`` to ``end_value``.

    Commonly used for the learning rate and for DINO's weight decay.

    Parameters
    ----------
    start_value : float
        Target value at the end of warm-up (peak LR for learning rate).
    end_value : float
        Final value at ``total_steps - 1``.
    total_steps : int
        Total number of values to produce (usually ``epochs * steps_per_epoch``).
    warmup_steps : int
        Number of steps of linear warm-up from ``warmup_start_value`` to
        ``start_value``.
    warmup_start_value : float
        Initial value at step 0.
    """
    assert total_steps > 0
    warmup = np.linspace(warmup_start_value, start_value, warmup_steps) if warmup_steps > 0 \
            else np.array([], dtype=np.float32)
    iters = np.arange(total_steps - warmup_steps)
    cosine = end_value + 0.5 * (start_value - end_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )
    return np.concatenate([warmup, cosine]).astype(np.float32)


def momentum_schedule(
    base_momentum: float = 0.996,
    final_momentum: float = 1.0,
    total_steps: int = 10000,
) -> np.ndarray:
    """Cosine schedule for DINO's teacher momentum: start low, end at 1.0."""
    return cosine_schedule(base_momentum, final_momentum, total_steps, warmup_steps=0)


# -----------------------------------------------------------------------------
# Parameter freezing
# -----------------------------------------------------------------------------

def deactivate_requires_grad(model: nn.Module) -> None:
    """Set ``requires_grad=False`` on every parameter of ``model``.

    Convenience wrapper called once on the teacher at construction time.
    """
    for p in model.parameters():
        p.requires_grad = False


def cancel_gradients_last_layer(model: nn.Module, current_epoch: int, freeze_epochs: int) -> None:
    """Zero-out gradients of the DINO head's weight-normalised last layer during
    the first ``freeze_epochs`` epochs.

    This is the "norm_last_layer"/"freeze_last_layer" trick from the DINO
    reference implementation. It stabilises the very early stages of training
    by preventing the last layer from moving before the projection MLP has
    produced meaningful features.
    """
    if current_epoch >= freeze_epochs:
        return
    for name, p in model.named_parameters():
        if "last_layer" in name:
            p.grad = None
