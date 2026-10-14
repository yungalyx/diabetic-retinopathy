"""Build stage: read sources, preprocess images, deduplicate, assign splits, write the manifest."""

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import polars as pl

from retina_dr.labels import DEFAULT_BINARY_CUTOFF, binary_from_grade
from retina_dr.preprocess import DEFAULT_SIZE, hash_existing, process_image
from retina_dr.records import SOURCES, Record, Source
from retina_dr.sources import read_source
from retina_dr.split import SplitConfig, assign_split

log = logging.getLogger(__name__)

MANIFEST_COLUMNS: Final = (
    "path",
    "source",
    "image_id",
    "patient_id",
    "grade",
    "binary_label",
    "split",
    "dhash",
)


@dataclass(frozen=True, slots=True)
class BuildConfig:
    raw_dir: Path
    out_dir: Path
    sources: tuple[Source, ...] = SOURCES
    size: int = DEFAULT_SIZE
    binary_cutoff: int = DEFAULT_BINARY_CUTOFF
    split: SplitConfig = field(default_factory=SplitConfig)
    workers: int = 8


@dataclass(frozen=True, slots=True)
class _Job:
    src: Path
    dst: Path
    size: int


def _run_job(job: _Job) -> int:
    if job.dst.is_file():
        return hash_existing(job.dst)
    return process_image(job.src, job.dst, job.size)


def _read_all(config: BuildConfig) -> list[Record]:
    records: list[Record] = []
    for source in config.sources:
        root = config.raw_dir / source
        if not root.is_dir():
            log.warning("%s: no folder at %s, skipping", source, root)
            continue
        found = read_source(source, root)
        log.info("%s: %d labelled images", source, len(found))
        records.extend(found)
    return records


def _relative_path(record: Record) -> str:
    return f"images/{record.source}/{record.image_id}.jpg"


def _process(records: list[Record], config: BuildConfig) -> list[int]:
    jobs = [_Job(r.raw_path, config.out_dir / _relative_path(r), config.size) for r in records]
    if config.workers <= 1:
        return [_run_job(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=config.workers) as pool:
        return list(pool.map(_run_job, jobs, chunksize=64))


def _deduplicate(frame: pl.DataFrame, config: BuildConfig) -> pl.DataFrame:
    """Keep one row per image hash.

    The holdout source wins, so no holdout image leaks into training.
    """
    priority = [config.split.holdout_source, *SOURCES]
    rank = {source: priority.index(source) for source in SOURCES}
    kept = (
        frame.with_columns(
            pl.col("source").replace_strict(rank, return_dtype=pl.Int8).alias("_rank")
        )
        .sort("_rank", "image_id")
        .unique(subset="dhash", keep="first", maintain_order=True)
        .drop("_rank")
    )
    dropped = frame.height - kept.height
    if dropped:
        log.warning("dropped %d duplicate images", dropped)
    return kept


def build(config: BuildConfig) -> pl.DataFrame:
    """Run the build stage and write manifest.parquet into out_dir. Rerunning reuses images."""
    records = _read_all(config)
    hashes = _process(records, config)
    frame = pl.DataFrame(
        {
            "path": [_relative_path(r) for r in records],
            "source": [r.source for r in records],
            "image_id": [r.image_id for r in records],
            "patient_id": [r.patient_id for r in records],
            "grade": [r.grade for r in records],
            "binary_label": [
                binary_from_grade(r.label_grade, config.binary_cutoff) for r in records
            ],
            "split": [assign_split(r.source, r.patient_id, config.split) for r in records],
            "dhash": hashes,
        },
        schema={
            "path": pl.String,
            "source": pl.String,
            "image_id": pl.String,
            "patient_id": pl.String,
            "grade": pl.Int8,
            "binary_label": pl.Int8,
            "split": pl.String,
            "dhash": pl.UInt64,
        },
    )
    manifest = _deduplicate(frame, config).select(MANIFEST_COLUMNS)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_parquet(config.out_dir / "manifest.parquet")
    return manifest


def summarize(manifest: pl.DataFrame) -> pl.DataFrame:
    """Image counts per source and split, for a quick check after a build."""
    return (
        manifest.group_by("source", "split")
        .agg(
            pl.len().alias("images"), pl.col("binary_label").mean().round(3).alias("positive_rate")
        )
        .sort("source", "split")
    )
