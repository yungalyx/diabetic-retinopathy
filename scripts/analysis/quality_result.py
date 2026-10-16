"""Print the quality-head run result and check it on the baseline's DDR grade 4 failures.

Run on a CPU pod with the network volume at /workspace.
"""

import json
from pathlib import Path

import polars as pl

RUNS = Path("/workspace/runs")
RUN = RUNS / "exp" / "b3-e10-q"
BASELINE = RUNS / "b3-20261015-1520-603e880"
KEYS = (
    "split",
    "source",
    "images",
    "auc",
    "sens_op",
    "spec_op",
    "qwk_tuned",
    "sens_vtdr_op",
    "severe_undergraded",
    "yield",
    "reject_rate_pos",
    "reject_rate_neg",
    "quality_auc",
    "reject_recall",
    "accepted_images",
    "auc_accepted",
    "sens_op_accepted",
    "spec_op_accepted",
    "severe_undergraded_rejected",
)


def main() -> None:
    metrics = json.loads((RUN / "metrics.json").read_text())
    print("THRESHOLDS", metrics["best_epoch"], metrics["threshold"], metrics["quality_threshold"])
    for row in metrics["rows"]:
        print("ROW", {k: row.get(k) for k in KEYS})
    reject = float(metrics["quality_threshold"])
    predictions = pl.read_parquet(RUN / "predictions.parquet")
    baseline = pl.read_parquet(BASELINE / "predictions.parquet")
    failures = baseline.filter(
        (pl.col("split") == "holdout") & (pl.col("grade") == 4) & (pl.col("pred_grade") == 0)
    ).select("path")
    joined = failures.join(predictions, on="path", how="left")
    flagged = int((joined["reject_prob"] > reject).sum())
    referred = int((joined["prob"] > float(metrics["threshold"])).sum())
    print("BASELINE_121", failures.height, "rejected", flagged, "referred", referred)
    ddr4 = predictions.filter((pl.col("split") == "holdout") & (pl.col("grade") == 4))
    print("DDR4_REJECT_RATE", round(float((ddr4["reject_prob"] > reject).mean()), 4))
    for line in (RUNS.parent / "bootstrap.log").read_text().splitlines()[-4000:]:
        if any(k in line for k in ("DOWNLOAD_OK", "ACCESS_ERR", "quality labels", "FAILED")):
            print("LOG", line[:300])


if __name__ == "__main__":
    main()
