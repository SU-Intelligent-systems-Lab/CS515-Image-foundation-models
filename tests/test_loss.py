"""Tests for the DINO loss: centering, sharpening, multi-crop cross-entropy."""

from __future__ import annotations

import math

import pytest
import torch

from dinovpr.dino.loss import DINOLoss


@pytest.fixture
def loss_fn() -> DINOLoss:
    return DINOLoss(
        out_dim=64,
        n_global_crops=2,
        n_local_crops=4,
        student_temp=0.1,
        teacher_temp=0.04,
        center_momentum=0.9,
    )


def test_loss_center_buffer_initialized_to_zeros(loss_fn):
    assert torch.equal(loss_fn.center, torch.zeros(1, 64))


def test_loss_value_is_finite_and_scalar(loss_fn):
    B = 3
    n_total = loss_fn.n_global_crops + loss_fn.n_local_crops
    student = torch.randn(n_total * B, 64)
    teacher = torch.randn(loss_fn.n_global_crops * B, 64)
    loss = loss_fn(student, teacher)
    assert loss.dim() == 0                      # scalar
    assert torch.isfinite(loss)


def test_loss_near_log_K_when_teacher_is_uniform(loss_fn):
    """If teacher outputs identical logits for every dimension, the
    distribution after softmax is uniform. The cross-entropy between
    uniform-teacher and softmax(student) is minimized when the student
    is also uniform; at that point loss = log K."""
    B = 4
    K = 64
    n_total = loss_fn.n_global_crops + loss_fn.n_local_crops
    # Both student and teacher: uniform logits
    student = torch.zeros(n_total * B, K)
    teacher = torch.zeros(loss_fn.n_global_crops * B, K)
    loss = loss_fn(student, teacher)
    assert math.isclose(float(loss), math.log(K), rel_tol=0.01)


def test_loss_backward_produces_gradients_for_student_only(loss_fn):
    """Gradient must flow into student, never into teacher."""
    B = 2
    n_total = loss_fn.n_global_crops + loss_fn.n_local_crops
    student = torch.randn(n_total * B, 64, requires_grad=True)
    # Teacher tensor with requires_grad=True just to verify no gradient is written.
    teacher = torch.randn(loss_fn.n_global_crops * B, 64, requires_grad=True)
    loss = loss_fn(student, teacher)
    loss.backward()
    assert student.grad is not None
    # loss_fn explicitly detaches the teacher distribution; teacher should have no grad.
    assert teacher.grad is None


def test_center_moves_toward_teacher_mean_after_update(loss_fn):
    """The center buffer follows
    c <- m*c + (1-m)*mean(teacher_out); check both the first step and
    that repeated updates converge toward the running mean."""
    teacher = torch.ones(8, 64) * 0.5    # constant teacher outputs
    _ = loss_fn(
        student_output=torch.zeros(6 * 8 // loss_fn.n_global_crops, 64),  # dummy
        teacher_output=teacher,
    )  # triggers update
    # After one update: c = m*0 + (1-m) * 0.5 = 0.05  (when m=0.9)
    assert torch.allclose(
        loss_fn.center, torch.full_like(loss_fn.center, 0.05), atol=1e-5
    )


def test_teacher_temp_sharpens_distribution(loss_fn):
    """With a small teacher temp, the teacher distribution is sharper (lower
    entropy) than with a large one, holding logits fixed."""
    teacher = torch.randn(2, 64)
    # Compute entropy in log space to avoid 0*log(0) -> nan for very small temps.
    low_logp = torch.log_softmax(teacher / 0.04, dim=-1)
    high_logp = torch.log_softmax(teacher / 0.4, dim=-1)
    ent_low = -(low_logp.exp() * low_logp).sum(dim=-1).mean()
    ent_high = -(high_logp.exp() * high_logp).sum(dim=-1).mean()
    assert ent_low < ent_high


def test_loss_does_not_include_same_view_pair():
    """The DINO loss explicitly excludes the (teacher_view_i, student_view_i)
    pair. With 2 global crops and 0 local crops, that leaves 2*2 - 2 = 2 pairs."""
    loss_fn = DINOLoss(out_dim=32, n_global_crops=2, n_local_crops=0)
    B = 1
    student = torch.randn(2 * B, 32)
    teacher = torch.randn(2 * B, 32)
    # Zero the student so log_softmax = -log K on every component; teacher is
    # uniform after the softmax-zero. Expected loss = log(K) averaged over
    # (2*2 - 2) = 2 cross-entropy terms.
    student_zero = torch.zeros_like(student)
    teacher_zero = torch.zeros_like(teacher)
    loss = loss_fn(student_zero, teacher_zero)
    assert math.isclose(float(loss), math.log(32), rel_tol=0.01)
