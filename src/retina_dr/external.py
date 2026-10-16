"""External test sets (D-029, D-035): IDRiD, DeepDRiD, and Messidor-2.

These sets never enter training, val, or the DDR holdout. Each one gets split "external".
The build writes `<out>/manifest.parquet` and processed images under `<out>/images/<source>/`.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import polars as pl

from retina_dr.labels import DEFAULT_BINARY_CUTOFF
from retina_dr.preprocess import process_image
from retina_dr.quality import GOOD, REJECT
from retina_dr.sources import index_images, read_csv_checked

log = logging.getLogger(__name__)

EXTERNAL_SOURCES: Final = ("idrid", "deepdrid", "messidor2")
EXTERNAL_SPLIT: Final = "external"


@dataclass(frozen=True, slots=True)
class ExternalRecord:
    source: str
    image_id: str
    raw_path: Path
    eye_id: str
    patient_id: str
    grade: int | None
    """ICDR grade 0-4, or None when the image is ungradable."""
    quality: int | None
    """EyeQ-style code: GOOD or REJECT, or None when the source gives no quality label."""


def _csv_with_name(root: Path, fragment: str) -> Path:
    found = sorted(p for p in root.rglob("*.csv") if fragment.lower() in p.name.lower())
    if not found:
        raise FileNotFoundError(f"no CSV with '{fragment}' in its name under {root}")
    return found[0]


def read_idrid(root: Path) -> list[ExternalRecord]:
    """IDRiD Disease Grading, test split only (103 images). No patient or eye link."""
    labels = read_csv_checked(
        _csv_with_name(root, "Disease Grading_Testing Labels"),
        ["Image name", "Retinopathy grade"],
    )
    index = index_images(root)
    records: list[ExternalRecord] = []
    for name, grade in labels.select("Image name", "Retinopathy grade").iter_rows():
        stem = Path(str(name)).stem
        if stem in index:
            records.append(ExternalRecord("idrid", stem, index[stem], stem, stem, int(grade), None))
    return records


def read_deepdrid(root: Path) -> list[ExternalRecord]:
    """DeepDRiD regular fundus, Online-Challenge 1 and 2 evaluation set (400 images).

    Image IDs are <patient>_<l|r><n>, two images per eye. "Overall quality" 1 is good, 0 is
    not acceptable (D-035). Read from Challenge1_labels.xlsx and Challenge2_labels.xlsx.
    """
    folder = next(root.rglob("Online-Challenge1&2-Evaluation"), None)
    if folder is None:
        raise FileNotFoundError(f"no Online-Challenge1&2-Evaluation folder under {root}")
    grades = pl.read_excel(folder / "Challenge1_labels.xlsx").select(
        pl.col("image_id").cast(pl.Utf8), pl.col("DR_Levels").cast(pl.Int64)
    )
    quality = pl.read_excel(folder / "Challenge2_labels.xlsx").select(
        pl.col("image_id").cast(pl.Utf8), pl.col("Overall quality").cast(pl.Int64)
    )
    labels = grades.join(quality, on="image_id", how="left")
    index = index_images(folder)
    records: list[ExternalRecord] = []
    for image_id, grade, overall in labels.iter_rows():
        if image_id not in index:
            continue
        patient, eye = image_id.split("_", 1)
        code = None if overall is None else (GOOD if int(overall) == 1 else REJECT)
        records.append(
            ExternalRecord(
                "deepdrid", image_id, index[image_id], f"{patient}_{eye[0]}", patient, grade, code
            )
        )
    return records


def read_messidor2(root: Path) -> list[ExternalRecord]:
    """Messidor-2 images (ADCIS) with the Google Brain adjudicated grades (messidor_data.csv).

    Ungradable images get grade None and quality REJECT. One image per eye; no patient link.
    """
    labels = read_csv_checked(
        _csv_with_name(root, "messidor_data"),
        ["image_id", "adjudicated_dr_grade", "adjudicated_gradable"],
    )
    index = index_images(root)
    records: list[ExternalRecord] = []
    for name, grade, gradable in labels.select(
        "image_id", "adjudicated_dr_grade", "adjudicated_gradable"
    ).iter_rows():
        stem = Path(str(name)).stem
        if stem not in index:
            continue
        ok = str(gradable).strip() == "1" and grade not in (None, "")
        records.append(
            ExternalRecord(
                "messidor2",
                stem,
                index[stem],
                stem,
                stem,
                int(float(str(grade))) if ok else None,
                GOOD if ok else REJECT,
            )
        )
    return records


_READERS: Final = {"idrid": read_idrid, "deepdrid": read_deepdrid, "messidor2": read_messidor2}


def build_external(raw_dir: Path, out_dir: Path, size: int) -> pl.DataFrame:
    """Process every external source found under raw_dir/<source> and write the manifest."""
    rows: list[dict[str, object]] = []
    for source in EXTERNAL_SOURCES:
        root = raw_dir / source
        if not root.is_dir():
            log.info("external: %s not found under %s, skipped", source, raw_dir)
            continue
        records = _READERS[source](root)
        log.info("external: %s has %d labelled images", source, len(records))
        for record in records:
            path = f"images/{source}/{record.image_id}.jpg"
            process_image(record.raw_path, out_dir / path, size)
            rows.append(
                {
                    "path": path,
                    "source": source,
                    "image_id": record.image_id,
                    "eye_id": f"{source}/{record.eye_id}",
                    "patient_id": record.patient_id,
                    "grade": record.grade,
                    "binary_label": None
                    if record.grade is None
                    else int(record.grade >= DEFAULT_BINARY_CUTOFF),
                    "quality": record.quality,
                    "split": EXTERNAL_SPLIT,
                }
            )
    if not rows:
        raise FileNotFoundError(f"no external source found under {raw_dir}: {EXTERNAL_SOURCES}")
    manifest = pl.DataFrame(
        rows,
        schema_overrides={"grade": pl.Int8, "binary_label": pl.Int8, "quality": pl.Int8},
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_parquet(out_dir / "manifest.parquet")
    return manifest
