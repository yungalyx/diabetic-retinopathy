"""Compare two decision rules for the quality head on finished predictions (D-032).

Rule A (quality first): reject when reject_prob is above the quality threshold.
Rule B (findings first): refer when prob is above the referral threshold, whatever the quality.
Reject only the images that rule B would call "not referable".
Thresholds come from metrics.json (set on val). Run on a CPU pod with the volume at /workspace.
"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl


def summarize(name: str, frame: pl.DataFrame, refer_t: float, reject_t: float) -> None:
    y = frame["binary_label"].to_numpy() == 1
    referred = frame["prob"].to_numpy() > refer_t
    low_quality = frame["reject_prob"].to_numpy() > reject_t
    for rule, rejected in (("A", low_quality), ("B", low_quality & ~referred)):
        decided = ~rejected
        missed = y & decided & ~referred
        print(
            "RULE",
            name,
            rule,
            {
                "images": len(y),
                "yield": round(float(decided.mean()), 4),
                "reject_pos": round(float(rejected[y].mean()), 4) if y.any() else None,
                "reject_neg": round(float(rejected[~y].mean()), 4) if (~y).any() else None,
                "sens_decided": round(float(referred[y & decided].mean()), 4) if y.any() else None,
                "spec_decided": round(float((~referred)[~y & decided].mean()), 4)
                if (~y).any()
                else None,
                "positives_called_negative": int(missed.sum()),
                "positives_total": int(y.sum()),
            },
            flush=True,
        )


def main(run_dir: Path) -> None:
    metrics = json.loads((run_dir / "metrics.json").read_text())
    refer_t, reject_t = float(metrics["threshold"]), float(metrics["quality_threshold"])
    predictions = pl.read_parquet(run_dir / "predictions.parquet")
    for split in ("val", "test", "holdout"):
        part = predictions.filter(pl.col("split") == split)
        summarize(split, part, refer_t, reject_t)
        vtdr = part.filter(pl.col("grade") >= 3)
        if vtdr.height:
            referred = vtdr["prob"].to_numpy() > refer_t
            rejected = vtdr["reject_prob"].to_numpy() > reject_t
            called_negative = int((~referred & ~rejected).sum())
            print("VTDR", split, vtdr.height, "called_negative_rule_B", called_negative, flush=True)
    print("DONE", np.__version__)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
