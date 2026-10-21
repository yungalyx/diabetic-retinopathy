from pathlib import Path

import numpy as np
import polars as pl

from retina_dr.train.local_tuning import rehearse, summarize_rehearsal


def _eyes(n_patients: int, seed: int = 0) -> pl.DataFrame:
    """Two eyes per patient; referable eyes score higher, but every score is shifted up."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        for side in ("l", "r"):
            label = int(rng.random() < 0.4)
            prob = float(np.clip(0.5 + 0.3 * label + rng.normal(0, 0.12), 0, 1))
            rows.append(
                {
                    "eye_id": f"p{p}_{side}",
                    "patient_id": f"p{p}",
                    "binary_label": label,
                    "prob": prob,
                    "reject_prob": 0.0,
                }
            )
    return pl.DataFrame(rows)


def test_rehearsal_keeps_patients_on_one_side_of_each_split() -> None:
    eyes = _eyes(40)
    rows = rehearse(eyes, n_splits=5, frozen_threshold=0.04, reject_threshold=0.97, seed=1)
    assert rows.height == 5
    assert (rows["tune_eyes"] + rows["test_eyes"] == 80).all()
    assert (rows["tune_eyes"] % 2 == 0).all()


def test_local_threshold_restores_specificity_that_a_low_frozen_threshold_loses() -> None:
    eyes = _eyes(200)
    rows = rehearse(eyes, n_splits=50, frozen_threshold=0.04, reject_threshold=0.97, seed=2)
    assert np.max(rows["frozen_spec"].to_numpy()) < 0.05
    assert abs(np.median(rows["local_spec"].to_numpy()) - 0.85) < 0.06
    assert np.median(rows["local_sens"].to_numpy()) > 0.8


def test_rejected_eyes_leave_the_decided_set() -> None:
    eyes = _eyes(30).with_columns(
        pl.when(pl.col("binary_label") == 0).then(0.99).otherwise(0.0).alias("reject_prob")
    )
    rows = rehearse(eyes, n_splits=3, frozen_threshold=0.99, reject_threshold=0.97, seed=3)
    # Every healthy eye is low quality and not referred at the frozen threshold: no result.
    assert rows["frozen_spec"].is_nan().all()


def test_summary_reports_median_range_and_pass_share() -> None:
    eyes = _eyes(200)
    rows = rehearse(eyes, n_splits=40, frozen_threshold=0.04, reject_threshold=0.97, seed=4)
    summary = summarize_rehearsal(rows, min_sens=0.85, min_spec=0.80)
    assert set(summary) >= {"local_spec_median", "local_spec_p05", "local_pass_share"}
    assert 0.0 <= summary["local_pass_share"] <= 1.0
    assert summary["frozen_pass_share"] == 0.0


def test_cli_writes_a_summary_per_source(tmp_path: Path) -> None:
    import json

    from retina_dr.cli import main

    images = _eyes(30).with_columns(pl.lit("deepdrid").alias("source"), pl.lit(2).alias("grade"))
    images.write_parquet(tmp_path / "predictions.parquet")
    frozen = {"threshold": 0.04, "quality_threshold": 0.97}
    (tmp_path / "metrics.json").write_text(json.dumps(frozen))
    code = main(
        [
            "rehearse-tuning",
            "--predictions",
            str(tmp_path / "predictions.parquet"),
            "--frozen-metrics",
            str(tmp_path / "metrics.json"),
            "--out-dir",
            str(tmp_path / "out"),
            "--splits",
            "4",
        ]
    )
    assert code == 0
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["deepdrid"]["splits"] == 4.0
