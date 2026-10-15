import numpy as np
import pytest

from retina_dr.train.sampling import rebalance_alpha, sample_weights, strata_from_labels


def test_alpha_zero_gives_uniform_weights() -> None:
    strata = np.array([0, 0, 0, 1])
    assert np.allclose(sample_weights(strata, 0.0), 0.25)


def test_alpha_one_gives_each_stratum_equal_total_weight() -> None:
    strata = np.array([0, 0, 0, 1, 2, 2])
    w = sample_weights(strata, 1.0)
    totals = [w[strata == s].sum() for s in (0, 1, 2)]
    assert np.allclose(totals, 1 / 3)


def test_weights_sum_to_one() -> None:
    strata = np.random.default_rng(0).integers(0, 7, 500)
    assert sample_weights(strata, 0.5).sum() == pytest.approx(1.0)


def test_alpha_decays_linearly_to_zero_at_the_last_epoch() -> None:
    assert rebalance_alpha(0, 5, 0.5) == pytest.approx(0.5)
    assert rebalance_alpha(2, 5, 0.5) == pytest.approx(0.25)
    assert rebalance_alpha(4, 5, 0.5) == pytest.approx(0.0)
    assert rebalance_alpha(0, 1, 0.5) == pytest.approx(0.0)


def test_strata_use_grade_when_present_and_binary_label_otherwise() -> None:
    grade = np.array([0, 3, -1, -1])
    binary = np.array([0, 1, 0, 1])
    assert strata_from_labels(grade, binary).tolist() == [0, 3, 5, 6]
