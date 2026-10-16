"""Metrics per split and per source from a predictions frame (D-009)."""

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from numpy.typing import NDArray

from retina_dr.quality import REJECT
from retina_dr.train.metrics import (
    DEFAULT_CUTPOINTS,
    apply_cutpoints,
    cluster_bootstrap_ci,
    quadratic_weighted_kappa,
    rates_at_threshold,
    roc_auc,
    sensitivity_at_specificity,
    threshold_at_specificity,
    tune_cutpoints,
    youden_threshold,
)

SPECIFICITIES: Final = (0.90, 0.95)
OPERATING_SPECIFICITY: Final = 0.85
"""Val specificity that sets the operating threshold (D-017). It leaves a margin above 0.80."""
VTDR_GRADE: Final = 3
"""Vision-threatening DR: grade 3 or higher."""
SEVERE_UNDERGRADE_MAX: Final = 1
"""A grade 3-4 image predicted at this grade or lower counts as severe undergrading."""
YIELD_TARGET: Final = 0.90
"""Share of val images that must get a result. The quality threshold respects it (D-025)."""
DEFAULT_RESAMPLES: Final = 500
ALL_SOURCES: Final = "all"

MetricRow = dict[str, float | int | str]
_NAN: Final = float("nan")


def _round(value: float) -> float:
    return value if math.isnan(value) else round(value, 4)


def _clusters(group: pl.DataFrame) -> NDArray[np.str_]:
    """Patient key for the bootstrap. Falls back to one cluster per row."""
    if "patient_id" in group.columns:
        keys = group.select(pl.concat_str("source", "patient_id", separator="/"))
        return keys.to_series().to_numpy().astype(np.str_)
    return np.arange(group.height).astype(np.str_)


def _share(mask: NDArray[np.bool_], within: NDArray[np.bool_]) -> float:
    return _round(float(mask[within].mean())) if within.any() else _NAN


def rejected_rows(
    frame: pl.DataFrame, threshold: float, reject_threshold: float
) -> NDArray[np.bool_]:
    """Findings-first rule (D-032): reject a low-quality image only when it is not referred."""
    low_quality = frame["reject_prob"].to_numpy().astype(np.float64) > reject_threshold
    referred = frame["prob"].to_numpy().astype(np.float64) > threshold
    rejected: NDArray[np.bool_] = low_quality & ~referred
    return rejected


def _quality_metrics(
    group: pl.DataFrame, threshold: float, reject_threshold: float
) -> dict[str, float | int]:
    """Yield, rejection by disease status, quality-head recall, and grading on accepted rows."""
    y = group["binary_label"].to_numpy().astype(np.int64)
    prob = group["prob"].to_numpy().astype(np.float64)
    rejected = rejected_rows(group, threshold, reject_threshold)
    quality = group["quality"].fill_null(-1).to_numpy().astype(np.int64)
    accepted = ~rejected
    sens, spec = rates_at_threshold(y[accepted], prob[accepted], threshold)
    labelled = quality >= 0
    return {
        "yield": _round(float(accepted.mean())) if group.height else _NAN,
        "reject_rate_pos": _share(rejected, y == 1),
        "reject_rate_neg": _share(rejected, y == 0),
        "quality_auc": _round(
            roc_auc(
                (quality[labelled] == REJECT).astype(np.int64),
                group["reject_prob"].to_numpy()[labelled],
            )
        ),
        "reject_recall": _share(rejected, quality == REJECT),
        "accepted_images": int(accepted.sum()),
        "auc_accepted": _round(roc_auc(y[accepted], prob[accepted])),
        "sens_op_accepted": _round(sens),
        "spec_op_accepted": _round(spec),
    }


def _metrics(
    group: pl.DataFrame,
    cuts: tuple[float, ...],
    threshold: float,
    reject_threshold: float | None,
    n_resamples: int,
    seed: int,
) -> dict[str, float | int]:
    y = group["binary_label"].to_numpy().astype(np.int64)
    prob = group["prob"].to_numpy().astype(np.float64)
    grade = group["grade"].fill_null(-1).to_numpy().astype(np.int64)
    graded = grade >= 0
    pred_grade = apply_cutpoints(group["severity"].to_numpy(), cuts)
    row: dict[str, float | int] = {
        "images": group.height,
        "positive_rate": _round(float(y.mean())) if group.height else _NAN,
        "auc": _round(roc_auc(y, prob)),
    }
    for spec in SPECIFICITIES:
        row[f"sens_at_spec{int(spec * 100)}"] = _round(sensitivity_at_specificity(y, prob, spec))
    sens, spec_op = rates_at_threshold(y, prob, threshold)
    row["sens_op"], row["spec_op"] = _round(sens), _round(spec_op)

    def qwk(idx: NDArray[np.int64]) -> float:
        keep = idx[graded[idx]]
        if len(keep) == 0:
            return _NAN
        return quadratic_weighted_kappa(grade[keep], pred_grade[keep])

    row["graded_images"] = int(graded.sum())
    if graded.any():
        row["qwk_fixed"] = _round(
            quadratic_weighted_kappa(
                grade[graded],
                apply_cutpoints(group["severity"].to_numpy()[graded], DEFAULT_CUTPOINTS),
            )
        )
        row["qwk_tuned"] = _round(qwk(np.arange(group.height, dtype=np.int64)))
    else:
        row["qwk_fixed"] = row["qwk_tuned"] = _NAN

    vtdr = grade >= VTDR_GRADE
    row["vtdr_images"] = int(vtdr.sum())
    row["sens_vtdr_op"] = _round(float((prob[vtdr] > threshold).mean())) if vtdr.any() else _NAN
    severe = vtdr & (pred_grade <= SEVERE_UNDERGRADE_MAX)
    row["severe_undergraded"] = int(severe.sum())
    row["severe_undergraded_rate"] = (
        _round(float(severe.sum() / vtdr.sum())) if vtdr.any() else _NAN
    )
    if reject_threshold is not None and "reject_prob" in group.columns:
        row.update(_quality_metrics(group, threshold, reject_threshold))
        rejected = rejected_rows(group, threshold, reject_threshold)
        row["severe_undergraded_rejected"] = int((severe & rejected).sum())

    statistics: dict[str, Callable[[NDArray[np.int64]], float]] = {
        "auc": lambda idx: roc_auc(y[idx], prob[idx]),
        "sens_op": lambda idx: rates_at_threshold(y[idx], prob[idx], threshold)[0],
        "spec_op": lambda idx: rates_at_threshold(y[idx], prob[idx], threshold)[1],
        "qwk_tuned": qwk,
    }
    clusters = _clusters(group)
    for name, statistic in statistics.items():
        low, high = cluster_bootstrap_ci(clusters, statistic, n_resamples, seed)
        row[f"{name}_lo"], row[f"{name}_hi"] = _round(low), _round(high)
    return row


def operating_threshold(predictions: pl.DataFrame, split: str = "val") -> float:
    """Probability threshold that gives OPERATING_SPECIFICITY on the given split."""
    chosen = predictions.filter(pl.col("split") == split)
    return threshold_at_specificity(
        chosen["binary_label"].to_numpy(), chosen["prob"].to_numpy(), OPERATING_SPECIFICITY
    )


def quality_threshold(predictions: pl.DataFrame, split: str = "val") -> float:
    """Reject threshold on the quality head, chosen on val only (D-025).

    Start from the Youden-optimal threshold on val rows with an EyeQ label. Raise it when
    needed, so that at least YIELD_TARGET of all val images get a result.
    """
    if "reject_prob" not in predictions.columns:
        return 1.0
    val = predictions.filter(pl.col("split") == split)
    if val.height == 0:
        return 1.0
    probs = np.sort(val["reject_prob"].to_numpy().astype(np.float64))
    # Lowest threshold that rejects at most (1 - YIELD_TARGET) of val: rows above it are rejected.
    allowed = math.floor((1.0 - YIELD_TARGET) * len(probs) + 1e-9)
    floor = float(probs[len(probs) - allowed - 1])
    labelled = val.filter(pl.col("quality").is_not_null())
    youden = youden_threshold(
        (labelled["quality"].to_numpy() == REJECT).astype(np.int64),
        labelled["reject_prob"].to_numpy(),
    )
    return floor if math.isnan(youden) else max(youden, floor)


def evaluate_predictions(
    predictions: pl.DataFrame,
    cuts: tuple[float, ...],
    threshold: float,
    quality_threshold: float | None = None,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> list[MetricRow]:
    """One row per split (source "all") and per split and source.

    Expects columns split, source, binary_label, grade (nullable), prob, severity.
    Uses patient_id for the bootstrap when present. Sets the operating point at `threshold`.
    With `quality_threshold` and a reject_prob column, adds the quality metrics (D-025).
    """
    rows: list[MetricRow] = []
    for split in sorted(predictions["split"].unique().to_list()):
        in_split = predictions.filter(pl.col("split") == split)
        groups = [(ALL_SOURCES, in_split)] + [
            (source, in_split.filter(pl.col("source") == source))
            for source in sorted(in_split["source"].unique().to_list())
        ]
        for source, group in groups:
            metrics = _metrics(group, cuts, threshold, quality_threshold, n_resamples, seed)
            rows.append({"split": split, "source": source, **metrics})
    return rows


def reevaluate_run(
    run_dir: Path, data_dir: Path, n_resamples: int = DEFAULT_RESAMPLES
) -> tuple[float, list[MetricRow]]:
    """Recompute metrics for a finished run from predictions.parquet. Does not load PyTorch.

    Adds patient_id from the manifest when the predictions do not have it.
    Reads the val-tuned cutpoints from metrics.json. Writes metrics_ci.json.
    """
    predictions = pl.read_parquet(run_dir / "predictions.parquet")
    if "patient_id" not in predictions.columns:
        manifest = pl.read_parquet(data_dir / "manifest.parquet", columns=["path", "patient_id"])
        predictions = predictions.join(manifest, on="path", how="left")
    saved = json.loads((run_dir / "metrics.json").read_text())
    cuts = tuple(float(c) for c in saved["cutpoints"])
    threshold = operating_threshold(predictions)
    reject = float(saved["quality_threshold"]) if "quality_threshold" in saved else None
    rows = evaluate_predictions(predictions, cuts, threshold, reject, n_resamples=n_resamples)
    result = {
        "cutpoints": list(cuts),
        "threshold": threshold,
        "operating_specificity": OPERATING_SPECIFICITY,
        "resamples": n_resamples,
        "rows": rows,
    }
    (run_dir / "metrics_ci.json").write_text(json.dumps(result, indent=2))
    return threshold, rows


_KEY_COLUMNS: Final = (
    "path",
    "source",
    "image_id",
    "patient_id",
    "split",
    "binary_label",
    "grade",
    "quality",
)
_MEAN_COLUMNS: Final = ("logit", "severity", "reject_prob")


def ensemble_runs(
    run_dirs: list[Path],
    out_dir: Path,
    n_resamples: int = DEFAULT_RESAMPLES,
    quality_run: Path | None = None,
) -> tuple[float, list[MetricRow]]:
    """Average the predictions of finished runs, then tune and evaluate as one model (D-026).

    Averages the binary logit, the severity value, and reject_prob when every run has it.
    With `quality_run`, reject_prob comes from that run only (D-033).
    Tunes cutpoints and thresholds on val only. Writes predictions.parquet and metrics.json.
    """
    frames = [pl.read_parquet(d / "predictions.parquet") for d in run_dirs]
    keys = [c for c in _KEY_COLUMNS if all(c in f.columns for f in frames)]
    means = [c for c in _MEAN_COLUMNS if all(c in f.columns for f in frames)]
    if quality_run is not None:
        means = [c for c in means if c != "reject_prob"]
    stacked = pl.concat([f.select(*keys, *means) for f in frames], how="vertical_relaxed")
    averaged = stacked.group_by(keys, maintain_order=True).agg(pl.col(means).mean())
    order = frames[0].select("path").with_row_index("_order")
    predictions = averaged.join(order, on="path").sort("_order").drop("_order")
    if quality_run is not None:
        rejects = pl.read_parquet(quality_run / "predictions.parquet").select("path", "reject_prob")
        predictions = predictions.join(rejects, on="path", how="left", maintain_order="left")
        if "quality" not in predictions.columns:
            quality = pl.read_parquet(quality_run / "predictions.parquet").select("path", "quality")
            predictions = predictions.join(quality, on="path", how="left", maintain_order="left")
    predictions = predictions.with_columns(
        pl.Series("prob", sigmoid(predictions["logit"].to_numpy().astype(np.float64)))
    )
    val = predictions.filter((pl.col("split") == "val") & pl.col("grade").is_not_null())
    cuts = (
        tune_cutpoints(val["severity"].to_numpy(), val["grade"].to_numpy())
        if val.height
        else DEFAULT_CUTPOINTS
    )
    predictions = predictions.with_columns(
        pl.Series("pred_grade", apply_cutpoints(predictions["severity"].to_numpy(), cuts))
    )
    threshold = operating_threshold(predictions)
    has_reject = "reject_prob" in predictions.columns
    reject = quality_threshold(predictions) if has_reject else None
    rows = evaluate_predictions(predictions, cuts, threshold, reject, n_resamples=n_resamples)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.write_parquet(out_dir / "predictions.parquet")
    result = {
        "members": [str(d) for d in run_dirs],
        "quality_run": None if quality_run is None else str(quality_run),
        "cutpoints": list(cuts),
        "threshold": threshold,
        "quality_threshold": reject,
        "rows": rows,
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    return threshold, rows


def sigmoid(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    result: NDArray[np.float64] = 1.0 / (1.0 + np.exp(-logits))
    return result
