import shutil
from pathlib import Path

import pytest

from retina_dr.records import Record
from retina_dr.sources import SOURCE_READERS, MissingColumnsError, index_images, read_source
from tests.conftest import MakeImage, write_fundus


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


def test_eyepacs_ignores_the_sample_folder(raw_dir: Path) -> None:
    sample = raw_dir / "eyepacs" / "sample"
    sample.mkdir()
    shutil.copy(raw_dir / "eyepacs" / "train" / "10_left.jpeg", sample / "10_left.jpeg")
    records = _by_id(read_source("eyepacs", raw_dir / "eyepacs"))
    assert records["10_left"].raw_path.parent.name == "train"


def test_odir_ignores_the_preprocessed_copies(raw_dir: Path, make_image: MakeImage) -> None:
    make_image(raw_dir / "odir" / "preprocessed_images" / "1_left.jpg", 99)
    records = _by_id(read_source("odir", raw_dir / "odir"))
    assert records["1_left"].raw_path.parent.name == "Training Images"


def test_identical_duplicates_keep_one_copy(tmp_path: Path, make_image: MakeImage) -> None:
    first = make_image(tmp_path / "a" / "x.png", 1)
    (tmp_path / "b").mkdir()
    shutil.copy(first, tmp_path / "b" / "x.png")
    assert index_images(tmp_path) == {"x": tmp_path / "a" / "x.png"}


def test_different_files_with_one_name_are_dropped(tmp_path: Path, make_image: MakeImage) -> None:
    make_image(tmp_path / "a" / "x.png", 1)
    write_fundus(tmp_path / "b" / "x.png", 2, size=(80, 60))
    make_image(tmp_path / "a" / "y.png", 3)
    assert set(index_images(tmp_path)) == {"y"}


@pytest.mark.parametrize(("id_col", "grade_col"), [("id_code", "diagnosis"), ("image", "level")])
def test_ddr_reads_grades_from_a_csv_when_no_txt_lists_exist(
    tmp_path: Path, make_image: MakeImage, id_col: str, grade_col: str
) -> None:
    root = tmp_path / "ddr"
    make_image(root / "DR_grading" / "DR_grading" / "a1.jpg", 1)
    make_image(root / "DR_grading" / "DR_grading" / "b2.jpg", 2)
    (root / "DR_grading.csv").write_text(f"{id_col},{grade_col}\na1.jpg,3\nb2,0\n")
    records = _by_id(read_source("ddr", root))
    assert {k: r.grade for k, r in records.items()} == {"a1": 3, "b2": 0}


def test_ddr_error_names_the_label_formats_it_accepts(
    tmp_path: Path, make_image: MakeImage
) -> None:
    make_image(tmp_path / "ddr" / "x.jpg", 1)
    (tmp_path / "ddr" / "labels.csv").write_text("name,score\nx,1\n")
    with pytest.raises(FileNotFoundError, match="id_code"):
        read_source("ddr", tmp_path / "ddr")
