import math

import numpy as np
import pytest

from retina_dr.train.metrics import (
    DEFAULT_CUTPOINTS,
    apply_cutpoints,
    quadratic_weighted_kappa,
    roc_auc,
    sensitivity_at_specificity,
    tune_cutpoints,
)


def test_kappa_is_one_for_perfect_agreement() -> None:
    y = np.array([0, 1, 2, 3, 4, 2, 0])
    assert quadratic_weighted_kappa(y, y) == pytest.approx(1.0)


def test_kappa_penalises_large_errors_more_than_small_ones() -> None:
    y = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    off_by_one = np.clip(y + 1, 0, 4)
    off_by_three = np.clip(y + 3, 0, 4)
    assert quadratic_weighted_kappa(y, off_by_one) > quadratic_weighted_kappa(y, off_by_three)


def test_kappa_is_near_zero_for_random_predictions() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 5000)
    assert abs(quadratic_weighted_kappa(y, rng.integers(0, 5, 5000))) < 0.05


def test_kappa_matches_a_hand_computed_value() -> None:
    # Two raters, 3 classes would differ; use the 5-class scale with a known table.
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    # O = [[1,1],[0,2]] on classes 0-1; weights (i-j)^2/16; E from marginals (2,2) x (1,3) / 4.
    observed = 1 * (1 / 16)
    expected = (2 * 3 / 4) * (1 / 16) + (2 * 1 / 4) * (1 / 16)
    assert quadratic_weighted_kappa(y_true, y_pred) == pytest.approx(1 - observed / expected)


def test_auc_is_one_for_perfect_ranking_and_half_for_ties() -> None:
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)


def test_auc_matches_pairwise_definition() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    s = rng.normal(size=300) + y
    pos, neg = s[y == 1], s[y == 0]
    pairs = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
    assert roc_auc(y, s) == pytest.approx(pairs)


def test_auc_is_nan_when_one_class_is_missing() -> None:
    assert math.isnan(roc_auc(np.array([1, 1]), np.array([0.2, 0.3])))


def test_sensitivity_at_specificity_uses_the_negative_score_quantile() -> None:
    y = np.array([0] * 10 + [1] * 4)
    s = np.array([*range(10), 8.5, 9.5, 10, 11], dtype=float)
    # Specificity 0.9 puts the threshold at the 9th negative score (8): 4 of 4 positives are above.
    assert sensitivity_at_specificity(y, s, 0.9) == pytest.approx(1.0)
    # Specificity 1.0 puts it at 9: 3 of 4 positives are above.
    assert sensitivity_at_specificity(y, s, 1.0) == pytest.approx(0.75)


def test_apply_cutpoints_maps_values_to_grades() -> None:
    values = np.array([-1.0, 0.49, 0.5, 1.7, 3.49, 3.5, 9.0])
    assert apply_cutpoints(values, DEFAULT_CUTPOINTS).tolist() == [0, 0, 1, 2, 3, 4, 4]


def test_tuned_cutpoints_fix_a_systematic_shift() -> None:
    rng = np.random.default_rng(2)
    grades = rng.integers(0, 5, 2000)
    values = grades + 0.45 + rng.normal(scale=0.2, size=2000)
    fixed = quadratic_weighted_kappa(grades, apply_cutpoints(values, DEFAULT_CUTPOINTS))
    cuts = tune_cutpoints(values, grades)
    tuned = quadratic_weighted_kappa(grades, apply_cutpoints(values, cuts))
    assert tuned > fixed + 0.02
    assert list(cuts) == sorted(cuts)
