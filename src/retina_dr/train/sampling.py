"""Class rebalancing that decays to the natural distribution during training (D-010)."""

from typing import Final

import numpy as np
from numpy.typing import NDArray

NO_GRADE: Final = -1
"""Grade value for rows without a clinician grade (ODIR)."""

_BINARY_STRATUM_OFFSET: Final = 5


def strata_from_labels(grade: NDArray[np.int64], binary: NDArray[np.int64]) -> NDArray[np.int64]:
    """Stratum per row: the grade 0-4 when present, else 5 + binary label."""
    strata: NDArray[np.int64] = np.where(grade >= 0, grade, _BINARY_STRATUM_OFFSET + binary)
    return strata.astype(np.int64)


def sample_weights(strata: NDArray[np.int64], alpha: float) -> NDArray[np.float64]:
    """Per-row sampling weights proportional to stratum_frequency ** -alpha, summing to 1.

    alpha 0 samples rows uniformly. alpha 1 gives every stratum equal total weight.
    """
    _, inverse, counts = np.unique(strata, return_inverse=True, return_counts=True)
    weights = counts[inverse].astype(np.float64) ** -alpha
    normalized: NDArray[np.float64] = weights / weights.sum()
    return normalized


def rebalance_alpha(epoch: int, epochs: int, start: float) -> float:
    """Linear decay from `start` at the first epoch to 0 at the last epoch."""
    if epochs <= 1:
        return 0.0
    return start * (1.0 - epoch / (epochs - 1))
