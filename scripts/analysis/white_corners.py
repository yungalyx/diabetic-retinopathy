"""Find processed images with white corners and compare model accuracy on them.

Run on a pod with the network volume at /workspace and PYTHONPATH=src:
    python scripts/analysis/white_corners.py TAR RUN_DIR

Reads every JPEG in the processed tar, measures the mean brightness of four 16x16 corner
patches, and flags an image when any corner is above 200 (0-255 scale).
Prints counts per source and split, metrics for flagged and other images, and a contact sheet.
"""

import base64
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image

from retina_dr.train.metrics import quadratic_weighted_kappa, rates_at_threshold, roc_auc

PATCH = 16
WHITE = 200
CHUNK = 8000


def corner_max(data: bytes) -> float:
    with Image.open(io.BytesIO(data)) as raw:
        gray = np.asarray(raw.convert("L"), dtype=np.float64)
    p = PATCH
    corners = [gray[:p, :p], gray[:p, -p:], gray[-p:, :p], gray[-p:, -p:]]
    return float(max(c.mean() for c in corners))


def scan(tar_path: Path) -> pl.DataFrame:
    rows = []
    with tarfile.open(tar_path) as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".jpg"):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            path = member.name.split("/", 1)[1]
            rows.append({"path": path, "corner_max": corner_max(handle.read())})
            if len(rows) % 10000 == 0:
                print("scanned", len(rows), flush=True)
    return pl.DataFrame(rows)


def group_metrics(frame: pl.DataFrame, threshold: float) -> dict[str, float | int]:
    y = frame["binary_label"].to_numpy()
    prob = frame["prob"].to_numpy()
    sens, spec = rates_at_threshold(y, prob, threshold)
    graded = frame.filter(pl.col("grade").is_not_null())
    qwk = (
        quadratic_weighted_kappa(graded["grade"].to_numpy(), graded["pred_grade"].to_numpy())
        if graded.height
        else float("nan")
    )
    return {
        "images": frame.height,
        "positives": int(y.sum()),
        "auc": round(roc_auc(y, prob), 4),
        "sens_op": round(sens, 4),
        "spec_op": round(spec, 4),
        "qwk": round(qwk, 4),
    }


def emit_sheet(tar_path: Path, paths: list[str]) -> None:
    wanted = set(paths)
    thumbs = []
    with tarfile.open(tar_path) as archive:
        for member in archive:
            name = member.name.split("/", 1)[-1]
            if name in wanted:
                handle = archive.extractfile(member)
                if handle is not None:
                    with Image.open(io.BytesIO(handle.read())) as raw:
                        thumbs.append(raw.convert("RGB").resize((128, 128)))
            if len(thumbs) == len(wanted):
                break
    canvas = Image.new("RGB", (4 * 128, ((len(thumbs) + 3) // 4) * 128), "black")
    for i, thumb in enumerate(thumbs):
        canvas.paste(thumb, ((i % 4) * 128, (i // 4) * 128))
    buffer = io.BytesIO()
    canvas.save(buffer, "JPEG", quality=60)
    text = base64.b64encode(buffer.getvalue()).decode()
    parts = [text[i : i + CHUNK] for i in range(0, len(text), CHUNK)]
    print(f"SHEET white_sample bytes={len(buffer.getvalue())} parts={len(parts)}", flush=True)
    for index, part in enumerate(parts):
        print(f"B64 white_sample {index:03d} {part}", flush=True)


def main(tar_path: Path, run_dir: Path) -> None:
    corners = scan(tar_path)
    corners.write_parquet(run_dir / "corner_scan.parquet")
    flagged = corners.with_columns((pl.col("corner_max") > WHITE).alias("white"))
    print("WHITE_TOTAL", int(flagged["white"].sum()), "of", flagged.height, flush=True)

    threshold = float(json.loads((run_dir / "metrics_ci.json").read_text())["threshold"])
    predictions = pl.read_parquet(run_dir / "predictions.parquet").join(
        flagged, on="path", how="left"
    )
    manifest = pl.read_parquet(tar_path.parent / "processed" / "manifest.parquet")
    by_source = (
        manifest.join(flagged, on="path", how="left")
        .group_by("source", "split")
        .agg(pl.len().alias("images"), pl.col("white").sum().alias("white"))
        .sort("source", "split")
    )
    for row in by_source.iter_rows(named=True):
        print("COUNT", row, flush=True)
    for (split, source), group in predictions.group_by("split", "source"):
        for white in (True, False):
            part = group.filter(pl.col("white") == white)
            if part.height:
                print("METRIC", split, source, white, group_metrics(part, threshold), flush=True)
    sample = predictions.filter(pl.col("white")).sample(
        min(16, int(predictions["white"].sum())), seed=0
    )
    for row in sample.select("path", "grade", "prob", "corner_max").iter_rows():
        print("SAMPLE", row, flush=True)
    if sample.height:
        emit_sheet(tar_path, sample["path"].to_list())
    print("ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
