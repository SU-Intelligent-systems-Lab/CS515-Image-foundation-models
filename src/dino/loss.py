"""
DINO loss: cross-entropy between sharpened teacher and softmaxed student
distributions, with an EMA-updated centering vector to prevent mode collapse.

Mathematical background
-----------------------
Given an image :math:`x` and a pair of augmented views, the student and
teacher produce :math:`K`-dimensional logits, converted to distributions via
temperature-softmax:

.. math::

    P_s(x)^{(k)} &= \\frac{\\exp(g_s(x)^{(k)}/\\tau_s)}{\\sum_j \\exp(g_s(x)^{(j)}/\\tau_s)} \\\\
    P_t(x)^{(k)} &= \\frac{\\exp((g_t(x)^{(k)} - c^{(k)})/\\tau_t)}{\\sum_j \\exp((g_t(x)^{(j)} - c^{(j)})/\\tau_t)}

The DINO loss is the cross-entropy between these:

.. math::

    L = -\\sum_k P_t(x_1)^{(k)} \\log P_s(x_2)^{(k)}

with :math:`\\tau_t \\ll \\tau_s` (sharpening the teacher) and the center
:math:`c` updated as an EMA over the batch to prevent the uniform-collapse
failure mode. Centering and sharpening exert opposite pressures on the
teacher's output entropy; their balance keeps training from collapsing.

Multi-crop
----------
When multi-crop augmentation is used (2 global + M local crops per image), the
student processes all crops and the teacher only the global ones. The full
loss sums cross-entropy over every (teacher-global, student-other) pair
``v != v_teacher``. See ``DINOLoss.forward`` for the exact summation.

References
----------
Caron et al., "Emerging Properties in Self-Supervised Vision Transformers",
ICCV 2021, Algorithm 1 and Section 3.
"""

from __future__ import annotations

from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    """DINO cross-entropy loss with centering + sharpening.

    Parameters
    ----------
    out_dim : int
        Prototype dimension :math:`K` (must match the DINO head output).
    n_global_crops : int
        Number of global crops per image fed to the teacher (DINO uses 2).
    n_local_crops : int
        Number of local crops per image fed to the student (DINO uses 6–8).
    student_temp : float
        Student softmax temperature :math:`\\tau_s` (typically 0.1).
    teacher_temp : float
        Teacher softmax temperature :math:`\\tau_t`. DINO warms this up over
        the first few epochs from 0.04 to 0.07; the warm-up must be handled
        by the caller, e.g. by calling ``loss_fn.teacher_temp = current_temp``
        before each forward.
    center_momentum : float
        EMA coefficient ``m`` in the center update
        ``c <- m*c + (1-m) * mean(g_t(x))``. DINO uses 0.9.
    """

    def __init__(
        self,
        out_dim: int,
        n_global_crops: int = 2,
        n_local_crops: int = 6,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.n_global_crops = n_global_crops
        self.n_local_crops = n_local_crops

        # Register the center as a buffer so it moves with .to(device) / save / load
        self.register_buffer("center", torch.zeros(1, out_dim))

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------
    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the DINO loss over all (teacher, student) crop pairs.

        Parameters
        ----------
        student_output : torch.Tensor, shape ``((Ng+Nl)*B, K)``
            Student logits over ALL crops, flattened batch-first. The expected
            crop order is: first ``Ng*B`` rows are the ``Ng`` global crops
            (one group per crop index, each of size B), followed by ``Nl*B``
            rows for the local crops.
        teacher_output : torch.Tensor, shape ``(Ng*B, K)``
            Teacher logits over GLOBAL crops only, same ordering convention.

        Returns
        -------
        loss : torch.Tensor, scalar
        """
        # --- Student: soft distribution ---
        # Split into one chunk per crop index (each chunk has B samples)
        student_out = student_output / self.student_temp
        student_out_list = student_out.chunk(self.n_global_crops + self.n_local_crops)

        # --- Teacher: center, sharpen, softmax, and detach ---
        # The detach() is crucial: gradient must not flow into the teacher.
        teacher_out = F.softmax(
            (teacher_output - self.center) / self.teacher_temp, dim=-1
        )
        teacher_out = teacher_out.detach().chunk(self.n_global_crops)

        # --- Multi-crop cross-entropy ---
        # For each (teacher global crop index iq, student crop index v) pair
        # with iq != v (DINO skips the identical view), add the cross-entropy.
        total_loss = 0.0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out_list)):
                if v == iq:
                    # Skip the teacher-vs-same-view case (it carries no learning signal
                    # beyond matching one's own EMA; DINO explicitly excludes it).
                    continue
                # Cross-entropy: -sum_k q_k * log_softmax(student)_k
                log_p = F.log_softmax(student_out_list[v], dim=-1)
                loss = torch.sum(-q * log_p, dim=-1)  # (B,)
                total_loss = total_loss + loss.mean()
                n_loss_terms += 1
        total_loss = total_loss / n_loss_terms

        # --- Update the center from the TEACHER's outputs ---
        # (Must be done after the softmax computation above, using the raw teacher logits.)
        self.update_center(teacher_output)

        return total_loss

    # -----------------------------------------------------------------
    # Center update
    # -----------------------------------------------------------------
    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor) -> None:
        """EMA update of the centering vector.

        .. math::

            c \\leftarrow m \\cdot c + (1-m) \\cdot \\mathrm{mean}_i g_t(x_i)

        Computed over all teacher outputs in the current batch (across all
        distributed workers, if a distributed process group is initialized).
        """
        batch_center = teacher_output.mean(dim=0, keepdim=True)  # (1, K)

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center = batch_center / dist.get_world_size()

        self.center = (
            self.center * self.center_momentum
            + batch_center * (1.0 - self.center_momentum)
        )
