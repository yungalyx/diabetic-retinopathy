"""Evaluation metrics in plain NumPy: kappa, ROC AUC, operating points, bootstrap, cutpoints."""

from collections.abc import Callable
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

N_GRADES: Final = 5
DEFAULT_CUTPOINTS: Final = (0.5, 1.5, 2.5, 3.5)
_TUNE_ROUNDS: Final = 3
_TUNE_STEPS: Final = 60
_MIN_GAP: Final = 0.05

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def quadratic_weighted_kappa(y_true: ArrayLike, y_pred: ArrayLike, n: int = N_GRADES) -> float:
    """Cohen's kappa with quadratic weights over grades 0..n-1."""
    t = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_pred, dtype=np.int64)
    observed = np.zeros((n, n), dtype=np.float64)
    np.add.at(observed, (t, p), 1.0)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / max(len(t), 1)
    idx = np.arange(n)
    weights = (idx[:, None] - idx[None, :]) ** 2 / (n - 1) ** 2
    denominator = float((weights * expected).sum())
    if denominator == 0.0:
        return 1.0 if np.array_equal(t, p) else 0.0
    return 1.0 - float((weights * observed).sum()) / denominator


def _average_ranks(values: FloatArray) -> FloatArray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    boundaries = np.flatnonzero(np.diff(sorted_values)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(values)]])
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        ranks[order[start:end]] = (start + end + 1) / 2.0
    return ranks


def roc_auc(y_true: ArrayLike, score: ArrayLike) -> float:
    """Area under the ROC curve (Mann-Whitney U, ties count half). NaN when one class is absent."""
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(score, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(s)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def threshold_at_specificity(y_true: ArrayLike, score: ArrayLike, specificity: float) -> float:
    """Lowest score threshold that keeps at least `specificity` of negatives at or below it.

    NaN when there are no negatives.
    """
    y = np.asarray(y_true, dtype=np.int64)
    negatives = np.sort(np.asarray(score, dtype=np.float64)[y == 0])
    if len(negatives) == 0:
        return float("nan")
    index = min(int(np.ceil(specificity * len(negatives))) - 1, len(negatives) - 1)
    return float(negatives[max(index, 0)])


def rates_at_threshold(
    y_true: ArrayLike, score: ArrayLike, threshold: float
) -> tuple[float, float]:
    """Sensitivity and specificity when scores above `threshold` count as positive."""
    y = np.asarray(y_true, dtype=np.int64)
    flagged = np.asarray(score, dtype=np.float64) > threshold
    positives, negatives = y == 1, y == 0
    sens = float(flagged[positives].mean()) if positives.any() else float("nan")
    spec = float((~flagged[negatives]).mean()) if negatives.any() else float("nan")
    return sens, spec


def youden_threshold(y_true: ArrayLike, score: ArrayLike) -> float:
    """Score threshold with the highest sensitivity + specificity - 1 (positive: score above).

    NaN when one class is missing.
    """
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(score, dtype=np.float64)
    positives, negatives = np.sort(s[y == 1]), np.sort(s[y == 0])
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")
    candidates = np.unique(s)
    tpr = 1.0 - np.searchsorted(positives, candidates, side="right") / len(positives)
    fpr = 1.0 - np.searchsorted(negatives, candidates, side="right") / len(negatives)
    return float(candidates[int(np.argmax(tpr - fpr))])


def sensitivity_at_specificity(y_true: ArrayLike, score: ArrayLike, specificity: float) -> float:
    """Share of positives above the score that keeps at least `specificity` of negatives below."""
    y = np.asarray(y_true, dtype=np.int64)
    if not (y == 1).any():
        return float("nan")
    threshold = threshold_at_specificity(y, score, specificity)
    if np.isnan(threshold):
        return float("nan")
    return rates_at_threshold(y, score, threshold)[0]


def cluster_bootstrap_ci(
    clusters: ArrayLike,
    statistic: Callable[[IntArray], float],
    n_resamples: int,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile interval of `statistic` over resamples of whole clusters (patients).

    `statistic` takes row indices. A resample draws clusters with replacement and keeps all
    rows of each drawn cluster. NaN when there are no resamples or every value is NaN.
    """
    if n_resamples <= 0:
        return float("nan"), float("nan")
    _, codes = np.unique(np.asarray(clusters), return_inverse=True)
    n_clusters = int(codes.max()) + 1 if len(codes) else 0
    if n_clusters == 0:
        return float("nan"), float("nan")
    rows = np.arange(len(codes), dtype=np.int64)
    rng = np.random.default_rng(seed)
    values = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        counts = np.bincount(rng.integers(0, n_clusters, n_clusters), minlength=n_clusters)
        values[i] = statistic(np.repeat(rows, counts[codes]))
    if np.isnan(values).all():
        return float("nan"), float("nan")
    tail = (1.0 - level) / 2.0 * 100.0
    low, high = np.nanpercentile(values, [tail, 100.0 - tail])
    return float(low), float(high)


def apply_cutpoints(values: ArrayLike, cuts: tuple[float, ...]) -> IntArray:
    """Map continuous severity values to grades: grade k when cuts[k-1] <= value < cuts[k]."""
    graded: IntArray = np.digitize(np.asarray(values, dtype=np.float64), cuts).astype(np.int64)
    return graded


def tune_cutpoints(
    values: ArrayLike, grades: ArrayLike, start: tuple[float, ...] = DEFAULT_CUTPOINTS
) -> tuple[float, ...]:
    """Coordinate search for the cutpoints that maximise quadratic weighted kappa."""
    v = np.asarray(values, dtype=np.float64)
    g = np.asarray(grades, dtype=np.int64)
    cuts = list(start)
    best = quadratic_weighted_kappa(g, apply_cutpoints(v, tuple(cuts)))
    for _ in range(_TUNE_ROUNDS):
        for i in range(len(cuts)):
            low = cuts[i - 1] + _MIN_GAP if i > 0 else cuts[i] - 1.0
            high = cuts[i + 1] - _MIN_GAP if i < len(cuts) - 1 else cuts[i] + 1.0
            for candidate in np.linspace(low, high, _TUNE_STEPS).tolist():
                trial = [*cuts[:i], candidate, *cuts[i + 1 :]]
                score = quadratic_weighted_kappa(g, apply_cutpoints(v, tuple(trial)))
                if score > best:
                    best, cuts = score, trial
    return tuple(round(c, 4) for c in cuts)
