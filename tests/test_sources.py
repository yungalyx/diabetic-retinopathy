from pathlib import Path

import pytest

from retina_dr.records import Record
from retina_dr.sources import SOURCE_READERS, MissingColumnsError, read_source


def _by_id(records: list[Record]) -> dict[str, Record]:
    return {r.image_id: r for r in records}


def test_eyepacs_groups_both_eyes_under_one_patient(raw_dir: Path) -> None:
    records = _by_id(read_source("eyepacs", raw_dir / "eyepacs"))
    assert records["10_left"].patient_id == records["10_right"].patient_id == "10"
    assert records["10_right"].grade == 3


def test_aptos_reads_grades_from_train_csv(raw_dir: Path) -> None:
    records = _by_id(read_source("aptos", raw_dir / "aptos"))
    assert {k: r.grade for k, r in records.items()} == {"abc123": 2, "def456": 4}


def test_ddr_drops_ungradable_images(raw_dir: Path) -> None:
    records = _by_id(read_source("ddr", raw_dir / "ddr"))
    assert set(records) == {"007-0001-000", "007-0003-000"}


def test_odir_emits_each_eye_once_and_keeps_grade_out_of_severity(raw_dir: Path) -> None:
    records = _by_id(read_source("odir", raw_dir / "odir"))
    assert set(records) == {"1_left", "1_right", "2_right"}
    assert records["1_right"].label_grade == 2
    assert all(r.grade is None for r in records.values())


def test_every_record_points_to_an_existing_image(raw_dir: Path) -> None:
    for source in SOURCE_READERS:
        for record in read_source(source, raw_dir / source):
            assert record.raw_path.is_file()


def test_records_without_an_image_file_are_skipped(raw_dir: Path) -> None:
    (raw_dir / "aptos" / "train_images" / "abc123.png").unlink()
    assert [r.image_id for r in read_source("aptos", raw_dir / "aptos")] == ["def456"]


def test_missing_label_column_fails_with_a_clear_error(raw_dir: Path) -> None:
    (raw_dir / "aptos" / "train.csv").write_text("id_code,label\nabc123,2\n")
    with pytest.raises(MissingColumnsError, match="diagnosis"):
        read_source("aptos", raw_dir / "aptos")
