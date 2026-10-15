import shutil
from pathlib import Path

import polars as pl
import pytest

from retina_dr.pipeline import MANIFEST_COLUMNS, BuildConfig, SourceReadError, build


@pytest.fixture
def config(raw_dir: Path, tmp_path: Path) -> BuildConfig:
    return BuildConfig(raw_dir=raw_dir, out_dir=tmp_path / "processed", size=32, workers=1)


def test_build_writes_manifest_with_expected_columns(config: BuildConfig) -> None:
    manifest = build(config)
    assert manifest.columns == list(MANIFEST_COLUMNS)
    assert pl.read_parquet(config.out_dir / "manifest.parquet").equals(manifest)


def test_every_manifest_path_exists_relative_to_out_dir(config: BuildConfig) -> None:
    manifest = build(config)
    assert all((config.out_dir / p).is_file() for p in manifest["path"])


def test_only_the_holdout_source_is_in_holdout(config: BuildConfig) -> None:
    manifest = build(config)
    holdout_sources = manifest.filter(pl.col("split") == "holdout")["source"].unique().to_list()
    assert holdout_sources == ["ddr"]


def test_binary_label_follows_the_cutoff(config: BuildConfig) -> None:
    manifest = build(config)
    eyepacs = dict(
        manifest.filter(pl.col("source") == "eyepacs")
        .select("image_id", "binary_label")
        .iter_rows()
    )
    assert eyepacs == {"10_left": 0, "10_right": 1, "13_left": 0}


def test_patient_eyes_share_a_split(config: BuildConfig) -> None:
    manifest = build(config)
    per_patient = manifest.group_by("source", "patient_id").agg(pl.col("split").n_unique())
    assert per_patient["split"].max() == 1


def test_duplicate_of_a_holdout_image_is_dropped_from_training_sources(
    config: BuildConfig, raw_dir: Path
) -> None:
    aptos_images = raw_dir / "aptos" / "train_images"
    (aptos_images / "abc123.png").unlink()
    shutil.copy(
        raw_dir / "ddr" / "DR_grading" / "test" / "007-0003-000.jpg", aptos_images / "abc123.jpg"
    )
    manifest = build(config)
    assert "abc123" not in manifest["image_id"].to_list()
    assert "007-0003-000" in manifest["image_id"].to_list()


def test_duplicates_with_conflicting_grades_are_all_dropped(
    config: BuildConfig, raw_dir: Path
) -> None:
    aptos = raw_dir / "aptos"
    (aptos / "train.csv").write_text("id_code,diagnosis\nabc123,0\ndef456,4\n")
    shutil.copy(aptos / "train_images" / "abc123.png", aptos / "train_images" / "def456.png")
    image_ids = build(config)["image_id"].to_list()
    assert "abc123" not in image_ids
    assert "def456" not in image_ids


def test_rebuild_reuses_processed_images(config: BuildConfig) -> None:
    build(config)
    sample = next((config.out_dir / "images").rglob("*.jpg"))
    before = sample.stat().st_mtime_ns
    build(config)
    assert sample.stat().st_mtime_ns == before


def test_missing_source_folder_is_skipped(config: BuildConfig, raw_dir: Path) -> None:
    shutil.rmtree(raw_dir / "odir")
    assert "odir" not in build(config)["source"].to_list()


def test_a_source_that_fails_to_parse_does_not_stop_the_others(
    config: BuildConfig, raw_dir: Path
) -> None:
    (raw_dir / "aptos" / "train.csv").write_text("wrong,columns\n1,2\n")
    with pytest.raises(SourceReadError, match="aptos"):
        build(config)
    manifest = pl.read_parquet(config.out_dir / "manifest.parquet")
    assert set(manifest["source"].unique()) == {"eyepacs", "ddr", "odir"}
