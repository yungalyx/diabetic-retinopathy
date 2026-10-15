"""Two-head loss: binary cross-entropy on every row, Huber on rows with a clinician grade."""

from dataclasses import dataclass

import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True, slots=True)
class LossParts:
    total: Tensor
    binary: Tensor
    severity: Tensor


def two_head_loss(
    binary_logit: Tensor,
    severity: Tensor,
    binary_target: Tensor,
    grade_target: Tensor,
    has_grade: Tensor,
    severity_weight: float = 1.0,
) -> LossParts:
    """Masked loss (D-002). Rows with has_grade 0 add nothing to the severity term."""
    bce = F.binary_cross_entropy_with_logits(binary_logit.float(), binary_target)
    per_row = F.huber_loss(severity.float(), grade_target, reduction="none", delta=1.0)
    huber = (per_row * has_grade).sum() / has_grade.sum().clamp(min=1.0)
    return LossParts(bce + severity_weight * huber, bce, huber)
