"""Model loss: grading terms on gradable rows, a quality term on rows with a quality label.

Binary cross-entropy and Huber skip rows that EyeQ marks as reject (D-025), because a grade
read from an ungradable image is not reliable. Huber also skips rows without a clinician grade.
"""

from dataclasses import dataclass

import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True, slots=True)
class LossParts:
    total: Tensor
    binary: Tensor
    severity: Tensor
    quality: Tensor


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


def model_loss(
    binary_logit: Tensor,
    severity: Tensor,
    quality_logit: Tensor,
    binary_target: Tensor,
    grade_target: Tensor,
    has_grade: Tensor,
    reject: Tensor,
    has_quality: Tensor,
    severity_weight: float = 1.0,
    quality_weight: float = 1.0,
) -> LossParts:
    """Masked loss (D-002, D-025). All targets and masks are float tensors of one row each."""
    gradable = 1.0 - reject * has_quality
    bce_rows = F.binary_cross_entropy_with_logits(
        binary_logit.float(), binary_target, reduction="none"
    )
    bce = _masked_mean(bce_rows, gradable)
    huber_rows = F.huber_loss(severity.float(), grade_target, reduction="none", delta=1.0)
    huber = _masked_mean(huber_rows, has_grade * gradable)
    quality_rows = F.binary_cross_entropy_with_logits(
        quality_logit.float(), reject, reduction="none"
    )
    quality = _masked_mean(quality_rows, has_quality)
    total = bce + severity_weight * huber + quality_weight * quality
    return LossParts(total, bce, huber, quality)
