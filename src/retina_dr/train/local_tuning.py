"""Rehearsal of local threshold tuning on a public set (D-041).

Split the eyes by patient into a tuning half and a test half. Set the referral threshold on the
tuning half (OPERATING_SPECIFICITY on non-referable eyes), then measure the test half. Compare
with the frozen val threshold on the same test half. Repeat over many random splits.
"""

import numpy as np
import polars as pl

from retina_dr.train.metrics import rates_at_threshold, threshold_at_specificity
from retina_dr.train.report import OPERATING_SPECIFICITY, rejected_rows


def _decided_rates(
    part: pl.DataFrame, threshold: float, reject_threshold: float
) -> tuple[float, float]:
    """Sensitivity and specificity on eyes that get a result (findings-first rule, D-032)."""
    decided = part.filter(~pl.Series(rejected_rows(part, threshold, reject_threshold)))
    if decided.height == 0:
        return float("nan"), float("nan")
    labels = decided["binary_label"].to_numpy().astype(np.int64)
    sens, spec = rates_at_threshold(labels, decided["prob"].to_numpy(), threshold)
    return float(sens), float(spec)


def rehearse(
    eyes: pl.DataFrame,
    n_splits: int,
    frozen_threshold: float,
    reject_threshold: float,
    tune_fraction: float = 0.5,
    target_specificity: float = OPERATING_SPECIFICITY,
    seed: int = 0,
) -> pl.DataFrame:
    """One row per random patient split: tuned threshold, then test rates for both thresholds."""
    rng = np.random.default_rng(seed)
    patients = np.array(sorted(eyes["patient_id"].unique().to_list()))
    rows: list[dict[str, float | int]] = []
    for split in range(n_splits):
        chosen = rng.permutation(patients)[: round(len(patients) * tune_fraction)]
        in_tune = eyes["patient_id"].is_in(chosen.tolist())
        tune, test = eyes.filter(in_tune), eyes.filter(~in_tune)
        tune_labels = tune["binary_label"].to_numpy().astype(np.int64)
        local = threshold_at_specificity(tune_labels, tune["prob"].to_numpy(), target_specificity)
        local_sens, local_spec = _decided_rates(test, local, reject_threshold)
        frozen_sens, frozen_spec = _decided_rates(test, frozen_threshold, reject_threshold)
        rows.append(
            {
                "split": split,
                "tune_eyes": tune.height,
                "test_eyes": test.height,
                "local_threshold": local,
                "local_sens": local_sens,
                "local_spec": local_spec,
                "frozen_sens": frozen_sens,
                "frozen_spec": frozen_spec,
            }
        )
    return pl.DataFrame(rows)


def summarize_rehearsal(rows: pl.DataFrame, min_sens: float, min_spec: float) -> dict[str, float]:
    """Median and 5th-95th percentile of each rate, and the share of splits that pass both."""
    summary: dict[str, float] = {"splits": float(rows.height)}
    for name in ("local", "frozen"):
        for rate in ("sens", "spec"):
            values = rows[f"{name}_{rate}"].drop_nans().to_numpy()
            if values.size == 0:
                values = np.array([np.nan])
            summary[f"{name}_{rate}_median"] = float(np.median(values))
            summary[f"{name}_{rate}_p05"] = float(np.percentile(values, 5))
            summary[f"{name}_{rate}_p95"] = float(np.percentile(values, 95))
        passed = (rows[f"{name}_sens"] >= min_sens) & (rows[f"{name}_spec"] >= min_spec)
        summary[f"{name}_pass_share"] = float(np.mean(passed.fill_null(False).to_numpy()))
    summary["local_threshold_median"] = float(np.nanmedian(rows["local_threshold"].to_numpy()))
    return summary
