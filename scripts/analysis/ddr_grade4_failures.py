"""Inspect DDR grade 4 images that the baseline predicts as grade 0.

Run on a pod with the network volume at /workspace:
    python scripts/analysis/ddr_grade4_failures.py RUN_DIR DATA_DIR

Prints group statistics and contact sheets as base64 JPEG lines ("B64 <sheet> <part> <data>"),
because the RunPod API gives no file transfer. Decode the lines to view the sheets.
"""

import base64
import io
import sys
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image, ImageDraw, ImageFilter

CHUNK = 8000
SEED = 0


def image_stats(path: Path) -> dict[str, float]:
    with Image.open(path) as raw:
        rgb = raw.convert("RGB").resize((256, 256))
    pixels = np.asarray(rgb, dtype=np.float64)
    gray = pixels.mean(axis=2)
    yy, xx = np.mgrid[:256, :256]
    disc = (yy - 127.5) ** 2 + (xx - 127.5) ** 2 < 120**2
    edges = np.asarray(rgb.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float64)
    inside = gray[disc]
    return {
        "brightness": float(inside.mean()),
        "contrast": float(inside.std()),
        "dark_frac": float((inside < 25).mean()),
        "bright_frac": float((inside > 200).mean()),
        "red_ratio": float(pixels[..., 0][disc].mean() / max(pixels[..., 1][disc].mean(), 1.0)),
        "sharpness": float(edges[disc].var()),
    }


def sheet(paths: list[Path], labels: list[str], cell: int, cols: int) -> bytes:
    rows = (len(paths) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell, rows * cell), "black")
    draw = ImageDraw.Draw(canvas)
    for i, (path, label) in enumerate(zip(paths, labels, strict=True)):
        with Image.open(path) as raw:
            thumb = raw.convert("RGB").resize((cell, cell))
        x, y = (i % cols) * cell, (i // cols) * cell
        canvas.paste(thumb, (x, y))
        draw.text((x + 3, y + 2), label, fill="yellow")
    buffer = io.BytesIO()
    canvas.save(buffer, "JPEG", quality=55)
    return buffer.getvalue()


def emit(name: str, data: bytes) -> None:
    text = base64.b64encode(data).decode()
    parts = [text[i : i + CHUNK] for i in range(0, len(text), CHUNK)]
    print(f"SHEET {name} bytes={len(data)} parts={len(parts)}", flush=True)
    for index, part in enumerate(parts):
        print(f"B64 {name} {index:03d} {part}", flush=True)


def main(run_dir: Path, data_dir: Path) -> None:
    predictions = pl.read_parquet(run_dir / "predictions.parquet").filter(
        pl.col("split") == "holdout"
    )
    groups = {
        "fail_4as0": predictions.filter((pl.col("grade") == 4) & (pl.col("pred_grade") == 0)),
        "ok_4as4": predictions.filter((pl.col("grade") == 4) & (pl.col("pred_grade") == 4)),
        "ok_0as0": predictions.filter((pl.col("grade") == 0) & (pl.col("pred_grade") == 0)),
        "g3_any": predictions.filter(pl.col("grade") == 3),
    }
    rows = []
    for name, group in groups.items():
        sample = group.sample(min(group.height, 150), seed=SEED)
        stats = [image_stats(data_dir / p) for p in sample["path"].to_list()]
        frame = pl.DataFrame(stats)
        row: dict[str, float | int | str] = {"group": name, "images": group.height}
        row["prob_mean"] = round(float(group["prob"].mean() or 0.0), 4)
        row["prob_median"] = round(float(group["prob"].median() or 0.0), 4)
        row["severity_mean"] = round(float(group["severity"].mean() or 0.0), 3)
        for column in frame.columns:
            row[column] = round(float(frame[column].mean() or 0.0), 4)
        rows.append(row)
    print("STATS", flush=True)
    for row in rows:
        print("STAT", row, flush=True)

    fail = groups["fail_4as0"].sort("prob")
    print("FAIL_PROB_QUANTILES", fail["prob"].quantile(0.5), fail["prob"].max(), flush=True)
    for name, group in groups.items():
        days = group["image_id"].str.slice(0, 8).value_counts().sort("count", descending=True)
        print("DAYS", name, group.height, days.head(8).rows(), flush=True)

    shuffled = fail.sample(fraction=1.0, seed=SEED, shuffle=True)
    paths = [data_dir / p for p in shuffled["path"].to_list()]
    labels = [f"{i} p{prob:.2f}" for i, prob in enumerate(shuffled["prob"].to_list())]
    emit("A_fail_grid", sheet(paths[:64], labels[:64], 128, 8))
    emit("D_fail_detail", sheet(paths[:6], labels[:6], 336, 3))
    ok = (
        groups["ok_4as4"]
        .head(0)
        .vstack(groups["ok_4as4"].sample(min(16, groups["ok_4as4"].height), seed=SEED))
    )
    emit(
        "B_ok_grade4",
        sheet([data_dir / p for p in ok["path"].to_list()], ["ok"] * ok.height, 128, 4),
    )
    print("ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
