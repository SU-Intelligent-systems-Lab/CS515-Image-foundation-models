"""Tests for EMA teacher update and schedule helpers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from dinovpr.dino.teacher_student import (
    cosine_schedule,
    deactivate_requires_grad,
    momentum_schedule,
    update_teacher_ema,
)


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 2)

    def forward(self, x):
        return self.b(torch.relu(self.a(x)))


def test_deactivate_requires_grad_turns_off_all_params():
    m = _TinyNet()
    deactivate_requires_grad(m)
    for p in m.parameters():
        assert p.requires_grad is False


def test_ema_momentum_one_leaves_teacher_unchanged():
    student = _TinyNet()
    teacher = _TinyNet()
    deactivate_requires_grad(teacher)
    before = {n: p.clone() for n, p in teacher.named_parameters()}
    update_teacher_ema(student, teacher, momentum=1.0)
    for n, p in teacher.named_parameters():
        assert torch.equal(p, before[n])


def test_ema_momentum_zero_copies_student_to_teacher():
    student = _TinyNet()
    teacher = _TinyNet()
    deactivate_requires_grad(teacher)
    # Perturb student so they differ
    for p in student.parameters():
        p.data.add_(torch.randn_like(p))
    update_teacher_ema(student, teacher, momentum=0.0)
    for sp, tp in zip(student.parameters(), teacher.parameters()):
        assert torch.allclose(sp.data, tp.data)


def test_ema_half_averages_parameters():
    student = _TinyNet()
    teacher = _TinyNet()
    deactivate_requires_grad(teacher)
    # Set known values
    for p in student.parameters():
        p.data.fill_(1.0)
    for p in teacher.parameters():
        p.data.fill_(3.0)
    update_teacher_ema(student, teacher, momentum=0.5)
    for tp in teacher.parameters():
        # 0.5*3 + 0.5*1 = 2
        assert torch.allclose(tp.data, torch.full_like(tp.data, 2.0))


def test_cosine_schedule_length_matches_total_steps():
    s = cosine_schedule(start_value=1.0, end_value=0.0, total_steps=100, warmup_steps=10)
    assert len(s) == 100


def test_cosine_schedule_warmup_is_linear():
    s = cosine_schedule(start_value=1.0, end_value=0.0, total_steps=100, warmup_steps=5,
                        warmup_start_value=0.0)
    # s[0] = 0.0, s[4] close to 1.0 (end of warmup)
    assert s[0] == 0.0
    # The last warmup entry is slightly less than start_value because linspace includes it.
    assert s[4] == 1.0


def test_cosine_schedule_decays_to_end_value():
    s = cosine_schedule(start_value=1.0, end_value=0.1, total_steps=50, warmup_steps=0)
    # Discrete cosine samples at k/N never quite reach k=N, so the last point
    # sits slightly above end_value. Just check it's close.
    assert abs(float(s[-1]) - 0.1) < 5e-3
    # And strictly between start and end
    assert 0.1 <= float(s[-1]) < 1.0


def test_momentum_schedule_monotonic_to_final():
    m = momentum_schedule(base_momentum=0.9, final_momentum=1.0, total_steps=1000)
    assert m[0] >= 0.9 - 1e-6
    # Non-decreasing overall
    assert m[-1] >= m[0]
    # Within [0, 1]
    assert m.min() >= 0.0 and m.max() <= 1.0 + 1e-5
