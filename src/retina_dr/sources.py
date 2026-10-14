"""Readers that turn each source's Kaggle layout into Records (Strategy pattern via a registry).

Each reader finds images by file stem anywhere under the source root, so
small differences between Kaggle mirrors do not break it.
"""

import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Final

import polars as pl

from retina_dr.labels import MAX_GRADE, odir_eye_grade
from retina_dr.records import Record, Source

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS: Final = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff"})

Reader = Callable[[Path], list[Record]]


class MissingColumnsError(ValueError):
    """A label file does not have the columns its reader needs."""


def index_images(root: Path) -> dict[str, Path]:
    """Map image file stem to path for every image under root."""
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file():
            continue
        if path.stem in index:
            raise ValueError(f"duplicate image name {path.stem}: {index[path.stem]} and {path}")
        index[path.stem] = path
    return index


def read_csv_checked(path: Path, required: Iterable[str]) -> pl.DataFrame:
    """Read a CSV as strings and fail when a required column is missing."""
    frame = pl.read_csv(path, infer_schema=False)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise MissingColumnsError(f"{path} is missing columns {missing}; found {frame.columns}")
    return frame


def find_file(root: Path, name: str) -> Path | None:
    """Return the first file with this name under root, or None."""
    return next(iter(sorted(root.rglob(name))), None)


def _require_file(root: Path, name: str) -> Path:
    found = find_file(root, name)
    if found is None:
        raise FileNotFoundError(f"{name} not found under {root}")
    return found


def _build(
    source: Source,
    rows: Iterable[tuple[str, str, int, int | None]],
    index: dict[str, Path],
) -> list[Record]:
    """Turn (image_id, patient_id, label_grade, grade) rows into Records.

    Skips ungradable rows and rows with no image file, and logs the counts.
    """
    records: list[Record] = []
    seen: set[str] = set()
    ungradable = missing = 0
    for image_id, patient_id, label_grade, grade in rows:
        if image_id in seen:
            continue
        seen.add(image_id)
        if not 0 <= label_grade <= MAX_GRADE:
            ungradable += 1
            continue
        path = index.get(image_id)
        if path is None:
            missing += 1
            continue
        records.append(Record(source, image_id, path, patient_id, label_grade, grade))
    if ungradable or missing:
        log.warning("%s: skipped %d ungradable, %d without image", source, ungradable, missing)
    return records


def read_eyepacs(root: Path) -> list[Record]:
    """EyePACS: trainLabels.csv, plus retinopathy_solution.csv for test images when present."""
    frames = [read_csv_checked(_require_file(root, "trainLabels.csv"), ["image", "level"])]
    solution = find_file(root, "retinopathy_solution.csv")
    if solution is not None:
        frames.append(read_csv_checked(solution, ["image", "level"]).select("image", "level"))
    labels = pl.concat(frames, how="vertical")
    rows = (
        (str(image), str(image).split("_")[0], int(level), int(level))
        for image, level in labels.select("image", "level").iter_rows()
    )
    return _build("eyepacs", rows, index_images(root))


def read_aptos(root: Path) -> list[Record]:
    """APTOS 2019: train.csv. APTOS gives no patient ID, so each image is its own patient."""
    labels = read_csv_checked(_require_file(root, "train.csv"), ["id_code", "diagnosis"])
    rows = (
        (str(code), str(code), int(dx), int(dx))
        for code, dx in labels.select("id_code", "diagnosis").iter_rows()
    )
    return _build("aptos", rows, index_images(root))


def read_ddr(root: Path) -> list[Record]:
    """DDR: DR_grading/{train,valid,test}.txt with lines "<file> <grade>". Grade 5 is ungradable."""
    rows: list[tuple[str, str, int, int | None]] = []
    for split_file in sorted(root.rglob("*.txt")):
        for line in split_file.read_text().splitlines():
            parts = line.split()
            if len(parts) != 2 or not parts[1].isdigit():
                continue
            stem = Path(parts[0]).stem
            grade = int(parts[1])
            rows.append((stem, stem, grade, grade if grade <= MAX_GRADE else None))
    if not rows:
        raise FileNotFoundError(f"no DDR grading .txt files found under {root}")
    return _build("ddr", rows, index_images(root))


def read_odir(root: Path) -> list[Record]:
    """ODIR-5K: full_df.csv, one row per patient with both eyes. Severity grade stays None."""
    columns = [
        "ID",
        "Left-Fundus",
        "Right-Fundus",
        "Left-Diagnostic Keywords",
        "Right-Diagnostic Keywords",
    ]
    labels = read_csv_checked(_require_file(root, "full_df.csv"), columns)
    rows: list[tuple[str, str, int, int | None]] = []
    for pid, left, right, left_kw, right_kw in labels.select(columns).iter_rows():
        for fundus, keywords in ((left, left_kw), (right, right_kw)):
            grade = odir_eye_grade(str(keywords or ""))
            rows.append((Path(str(fundus)).stem, str(pid), grade, None))
    return _build("odir", rows, index_images(root))


SOURCE_READERS: Final[dict[Source, Reader]] = {
    "eyepacs": read_eyepacs,
    "aptos": read_aptos,
    "ddr": read_ddr,
    "odir": read_odir,
}


def read_source(source: Source, root: Path) -> list[Record]:
    """Read one source's records from its raw folder."""
    return SOURCE_READERS[source](root)
