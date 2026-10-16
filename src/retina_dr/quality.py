"""Image quality labels from EyeQ (Fu et al. 2019) for EyePACS images (D-025).

EyeQ codes: 0 good, 1 usable, 2 reject. Reject means the image cannot be graded.
"""

from pathlib import Path
from typing import Final

import polars as pl

GOOD: Final = 0
USABLE: Final = 1
REJECT: Final = 2
EYEQ_URL: Final = "https://raw.githubusercontent.com/HzFu/EyeQ/master/data/Label_EyeQ_train.csv"
"""EyeQ train labels. They cover 12,543 EyePACS training images (CC BY-NC-SA 4.0)."""


def read_eyeq(path: Path) -> pl.DataFrame:
    """Return (image_id, quality) with the file extension removed from the image name."""
    frame = pl.read_csv(path, infer_schema=False)
    missing = {"image", "quality"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}: needs image and quality")
    return frame.select(
        pl.col("image").str.replace(r"\.[^.]+$", "").alias("image_id"),
        pl.col("quality").cast(pl.Int8),
    ).unique(subset="image_id", keep="first")


def attach_quality(manifest: pl.DataFrame, eyeq: pl.DataFrame) -> pl.DataFrame:
    """Add or replace the nullable `quality` column. Only EyePACS rows can match."""
    base = manifest.drop("quality", strict=False)
    labels = eyeq.with_columns(pl.lit("eyepacs").alias("source"))
    return base.join(labels, on=["source", "image_id"], how="left", maintain_order="left")
