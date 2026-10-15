"""Build stage: read sources, preprocess images, deduplicate, assign splits, write the manifest."""

import itertools
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

from retina_dr.dedupe import DUPLICATE_RMS, Fingerprints, duplicate_groups, nearest_distances
from retina_dr.labels import DEFAULT_BINARY_CUTOFF, binary_from_grade
from retina_dr.preprocess import DEFAULT_SIZE, Pixels, fingerprint_existing, process_image
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
    "dup_group",
)

PROGRESS_EVERY: Final = 5000
DISTANCE_BINS: Final = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, float("inf"))


@dataclass(frozen=True, slots=True)
class BuildConfig:
    raw_dir: Path
    out_dir: Path
    sources: tuple[Source, ...] = SOURCES
    size: int = DEFAULT_SIZE
    binary_cutoff: int = DEFAULT_BINARY_CUTOFF
    split: SplitConfig = field(default_factory=SplitConfig)
    workers: int = 8
    max_duplicate_rms: float = DUPLICATE_RMS


@dataclass(frozen=True, slots=True)
class _Job:
    src: Path
    dst: Path
    size: int


def _run_job(job: _Job) -> Pixels:
    if job.dst.is_file():
        return fingerprint_existing(job.dst)
    return process_image(job.src, job.dst, job.size)


class SourceReadError(RuntimeError):
    """One or more sources failed to parse. The manifest holds the other sources."""


def _read_all(config: BuildConfig) -> tuple[list[Record], dict[Source, str]]:
    """Read every source. A source that fails is logged and reported, not fatal."""
    records: list[Record] = []
    failed: dict[Source, str] = {}
    for source in config.sources:
        root = config.raw_dir / source
        if not root.is_dir():
            log.warning("%s: no folder at %s, skipping", source, root)
            continue
        try:
            found = read_source(source, root)
        except (OSError, ValueError, pl.exceptions.PolarsError) as error:
            log.error("%s: failed to read labels: %s", source, error)
            failed[source] = str(error)
            continue
        log.info("%s: %d labelled images", source, len(found))
        records.extend(found)
    return records, failed


def _relative_path(record: Record) -> str:
    return f"images/{record.source}/{record.image_id}.jpg"


def _process(records: list[Record], config: BuildConfig) -> Fingerprints:
    """Preprocess every image (or reuse it) and return one fingerprint row per record."""
    jobs = [_Job(r.raw_path, config.out_dir / _relative_path(r), config.size) for r in records]
    results: list[Pixels] = []
    if config.workers <= 1:
        results = [_run_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as pool:
            for done, fp in enumerate(pool.map(_run_job, jobs, chunksize=64), start=1):
                results.append(fp)
                if done % PROGRESS_EVERY == 0:
                    log.info("processed %d of %d images", done, len(jobs))
    if not results:
        return np.zeros((0, 0), dtype=np.uint8)
    return np.stack(results)


def _log_distances(fingerprints: Fingerprints) -> None:
    """Log how far each image is from its nearest neighbour, to check the duplicate threshold."""
    if len(fingerprints) < 2:
        return
    counts, _ = np.histogram(nearest_distances(fingerprints), bins=DISTANCE_BINS)
    edges = [f"{lo:g}-{hi:g}" for lo, hi in itertools.pairwise(DISTANCE_BINS)]
    log.info("nearest-neighbour RMS distance: %s", dict(zip(edges, counts.tolist(), strict=True)))


def _deduplicate(frame: pl.DataFrame, config: BuildConfig) -> pl.DataFrame:
    """Keep one row per duplicate group.

    Groups whose labels disagree are all dropped, because no copy can be trusted.
    Otherwise the holdout source wins, so no holdout image leaks into training.
    """
    priority = [config.split.holdout_source, *SOURCES]
    rank = {source: priority.index(source) for source in SOURCES}
    conflicting = (
        frame.group_by("dup_group")
        .agg(
            pl.col("binary_label").n_unique().alias("_labels"),
            pl.col("grade").drop_nulls().n_unique().alias("_grades"),
        )
        .filter((pl.col("_labels") > 1) | (pl.col("_grades") > 1))
        .select("dup_group")
    )
    kept = (
        frame.join(conflicting, on="dup_group", how="anti")
        .with_columns(pl.col("source").replace_strict(rank, return_dtype=pl.Int8).alias("_rank"))
        .sort("_rank", "image_id")
        .unique(subset="dup_group", keep="first", maintain_order=True)
        .drop("_rank")
    )
    groups = frame.group_by("dup_group").len().filter(pl.col("len") > 1)
    log.info("%d duplicate groups covering %d images", groups.height, int(groups["len"].sum()))
    if conflicting.height:
        log.warning("dropped %d duplicate groups with conflicting labels", conflicting.height)
    if dropped := frame.height - kept.height:
        log.warning("dropped %d duplicate images in total", dropped)
    return kept


def build(config: BuildConfig) -> pl.DataFrame:
    """Run the build stage and write manifest.parquet into out_dir. Rerunning reuses images."""
    records, failed = _read_all(config)
    fingerprints = _process(records, config)
    if len(records):
        _log_distances(fingerprints)
        groups = duplicate_groups(fingerprints, config.max_duplicate_rms).tolist()
    else:
        groups = []
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
            "dup_group": groups,
        },
        schema={
            "path": pl.String,
            "source": pl.String,
            "image_id": pl.String,
            "patient_id": pl.String,
            "grade": pl.Int8,
            "binary_label": pl.Int8,
            "split": pl.String,
            "dup_group": pl.Int64,
        },
    )
    config.out_dir.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(config.out_dir / "all_records.parquet")
    manifest = _deduplicate(frame, config).select(MANIFEST_COLUMNS)
    manifest.write_parquet(config.out_dir / "manifest.parquet")
    if failed:
        raise SourceReadError(f"sources failed to read: {failed}")
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
