from pathlib import Path

import polars as pl
import pytest

from retina_dr.quality import REJECT, attach_quality, read_eyeq


def _eyeq_csv(path: Path) -> Path:
    path.write_text(
        ",image,quality,DR_grade\n0,10_left.jpeg,0,0\n1,10_right.jpeg,2,1\n2,99_left.jpeg,1,0\n"
    )
    return path


def test_read_eyeq_strips_the_extension_and_keeps_the_quality_code(tmp_path: Path) -> None:
    frame = read_eyeq(_eyeq_csv(tmp_path / "eyeq.csv"))
    assert dict(frame.iter_rows()) == {"10_left": 0, "10_right": REJECT, "99_left": 1}


def test_read_eyeq_rejects_a_file_without_the_quality_column(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("image,grade\n10_left.jpeg,0\n")
    with pytest.raises(ValueError, match="quality"):
        read_eyeq(path)


def test_attach_quality_labels_only_matching_eyepacs_rows(tmp_path: Path) -> None:
    manifest = pl.DataFrame(
        {
            "source": ["eyepacs", "eyepacs", "aptos", "eyepacs"],
            "image_id": ["10_left", "10_right", "10_left", "11_left"],
        }
    )
    out = attach_quality(manifest, read_eyeq(_eyeq_csv(tmp_path / "eyeq.csv")))
    assert out["quality"].to_list() == [0, REJECT, None, None]
    assert out.height == manifest.height


def test_attach_quality_replaces_an_existing_quality_column(tmp_path: Path) -> None:
    manifest = pl.DataFrame({"source": ["eyepacs"], "image_id": ["10_left"], "quality": [2]})
    out = attach_quality(manifest, read_eyeq(_eyeq_csv(tmp_path / "eyeq.csv")))
    assert out["quality"].to_list() == [0]
    assert out.columns.count("quality") == 1


def test_add_quality_command_updates_the_manifest(tmp_path: Path) -> None:
    from retina_dr.cli import main

    pl.DataFrame({"source": ["eyepacs"], "image_id": ["10_right"]}).write_parquet(
        tmp_path / "manifest.parquet"
    )
    code = main(
        [
            "add-quality",
            "--data-dir",
            str(tmp_path),
            "--eyeq-csv",
            str(_eyeq_csv(tmp_path / "eyeq.csv")),
        ]
    )
    assert code == 0
    assert pl.read_parquet(tmp_path / "manifest.parquet")["quality"].to_list() == [REJECT]
