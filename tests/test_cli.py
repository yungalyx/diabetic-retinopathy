from pathlib import Path

import polars as pl
import pytest

from retina_dr.cli import main, parse_sources


def test_build_command_writes_manifest_and_prints_summary(
    raw_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "processed"
    code = main(
        [
            "build",
            "--raw-dir",
            str(raw_dir),
            "--out-dir",
            str(out),
            "--size",
            "32",
            "--workers",
            "1",
        ]
    )
    assert code == 0
    assert pl.read_parquet(out / "manifest.parquet").height > 0
    assert "holdout" in capsys.readouterr().out


def test_parse_sources_rejects_an_unknown_source() -> None:
    with pytest.raises(SystemExit):
        parse_sources("eyepacs,messidor")


def test_parse_sources_keeps_the_given_order() -> None:
    assert parse_sources("odir,aptos") == ("odir", "aptos")


def test_train_command_runs_a_tiny_training(tmp_path: Path) -> None:
    from tests.test_training import TINY_MODEL, _processed_dir

    data = _processed_dir(tmp_path / "data")
    code = main(
        [
            "train",
            "--data-dir",
            str(data),
            "--run-dir",
            str(tmp_path / "run"),
            "--model",
            TINY_MODEL,
            "--no-pretrained",
            "--image-size",
            "32",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--workers",
            "0",
            "--device",
            "cpu",
            "--no-tta",
        ]
    )
    assert code == 0
    assert (tmp_path / "run" / "metrics.json").is_file()
