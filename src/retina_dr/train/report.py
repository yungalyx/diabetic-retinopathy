"""Metrics per split and per source from a predictions frame (D-009)."""

import math
from typing import Final

import numpy as np
import polars as pl

from retina_dr.train.metrics import (
    DEFAULT_CUTPOINTS,
    apply_cutpoints,
    quadratic_weighted_kappa,
    roc_auc,
    sensitivity_at_specificity,
)

SPECIFICITIES: Final = (0.90, 0.95)
ALL_SOURCES: Final = "all"

MetricRow = dict[str, float | int | str]


def _round(value: float) -> float:
    return value if math.isnan(value) else round(value, 4)


def _metrics(group: pl.DataFrame, cuts: tuple[float, ...]) -> dict[str, float | int]:
    y = group["binary_label"].to_numpy()
    prob = group["prob"].to_numpy()
    row: dict[str, float | int] = {
        "images": group.height,
        "positive_rate": _round(float(y.mean())) if group.height else float("nan"),
        "auc": _round(roc_auc(y, prob)),
    }
    for spec in SPECIFICITIES:
        row[f"sens_at_spec{int(spec * 100)}"] = _round(sensitivity_at_specificity(y, prob, spec))
    graded = group.filter(pl.col("grade").is_not_null())
    row["graded_images"] = graded.height
    if graded.height:
        truth = graded["grade"].to_numpy()
        values = graded["severity"].to_numpy()
        row["qwk_fixed"] = _round(
            quadratic_weighted_kappa(truth, apply_cutpoints(values, DEFAULT_CUTPOINTS))
        )
        row["qwk_tuned"] = _round(quadratic_weighted_kappa(truth, apply_cutpoints(values, cuts)))
    else:
        row["qwk_fixed"] = row["qwk_tuned"] = float("nan")
    return row


def evaluate_predictions(predictions: pl.DataFrame, cuts: tuple[float, ...]) -> list[MetricRow]:
    """One row per split (source "all") and per split and source.

    Expects columns split, source, binary_label, grade (nullable), prob, severity.
    """
    rows: list[MetricRow] = []
    for split in sorted(predictions["split"].unique().to_list()):
        in_split = predictions.filter(pl.col("split") == split)
        rows.append({"split": split, "source": ALL_SOURCES, **_metrics(in_split, cuts)})
        for source in sorted(in_split["source"].unique().to_list()):
            group = in_split.filter(pl.col("source") == source)
            rows.append({"split": split, "source": source, **_metrics(group, cuts)})
    return rows


def sigmoid(logits: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
    result: np.typing.NDArray[np.float64] = 1.0 / (1.0 + np.exp(-logits))
    return result
