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


def test_evaluate_command_recomputes_metrics_with_patient_ids(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    data = tmp_path / "data"
    data.mkdir()
    pl.DataFrame(
        {"path": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"], "patient_id": ["1", "1", "2", "3"]}
    ).write_parquet(data / "manifest.parquet")
    run = tmp_path / "run"
    run.mkdir()
    pl.DataFrame(
        {
            "path": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
            "source": ["eyepacs"] * 4,
            "split": ["val", "val", "holdout", "holdout"],
            "binary_label": [0, 1, 0, 1],
            "grade": [0, 3, 1, 4],
            "prob": [0.1, 0.9, 0.2, 0.8],
            "severity": [0.1, 3.0, 1.0, 3.9],
        }
    ).write_parquet(run / "predictions.parquet")
    (run / "metrics.json").write_text(json.dumps({"cutpoints": [0.5, 1.5, 2.5, 3.5]}))
    code = main(["evaluate", "--run-dir", str(run), "--data-dir", str(data), "--resamples", "10"])
    assert code == 0
    result = json.loads((run / "metrics_ci.json").read_text())
    assert result["threshold"] == 0.1
    holdout = next(r for r in result["rows"] if r["split"] == "holdout")
    assert holdout["sens_op"] == 1.0
    assert "holdout" in capsys.readouterr().out


def test_ensemble_command_averages_runs_and_writes_metrics(tmp_path: Path) -> None:
    import json

    rows = {
        "path": ["a", "b", "c", "d", "e", "f"],
        "source": ["eyepacs"] * 6,
        "patient_id": ["1", "2", "3", "4", "5", "6"],
        "split": ["val", "val", "val", "holdout", "holdout", "holdout"],
        "binary_label": [0, 1, 0, 0, 1, 1],
        "grade": [0, 3, 1, 0, 2, 4],
    }
    runs = []
    for index, shift in enumerate((-1.0, 1.0)):
        run = tmp_path / f"run{index}"
        run.mkdir()
        logits = [-2.0 + shift, 2.0 + shift, -1.0 + shift, -2.0 + shift, 1.0 + shift, 3.0 + shift]
        pl.DataFrame(
            {
                **rows,
                "logit": logits,
                "severity": [0.0 + shift, 3.0, 1.0, 0.0, 2.0, 4.0 - shift],
            }
        ).write_parquet(run / "predictions.parquet")
        runs.append(str(run))
    out = tmp_path / "ensemble"
    code = main(["ensemble", "--run-dirs", *runs, "--out-dir", str(out), "--resamples", "0"])
    assert code == 0
    predictions = pl.read_parquet(out / "predictions.parquet")
    assert predictions["logit"].to_list() == [-2.0, 2.0, -1.0, -2.0, 1.0, 3.0]
    assert predictions["severity"].to_list()[0] == 0.0
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["members"] == runs
    assert any(r["split"] == "holdout" for r in metrics["rows"])


def test_ensemble_takes_the_reject_probability_from_the_quality_run(tmp_path: Path) -> None:
    import json

    base = {
        "path": ["a", "b", "c", "d"],
        "source": ["eyepacs"] * 4,
        "patient_id": ["1", "2", "3", "4"],
        "split": ["val", "val", "holdout", "holdout"],
        "binary_label": [0, 1, 0, 1],
        "grade": [0, 3, 0, 4],
        "quality": [0, 2, None, None],
    }
    dr = tmp_path / "dr"
    dr.mkdir()
    pl.DataFrame(
        {**base, "logit": [-2.0, 2.0, -2.0, 2.0], "severity": [0.0, 3.0, 0.0, 4.0]},
        schema_overrides={"quality": pl.Int8},
    ).write_parquet(dr / "predictions.parquet")
    quality = tmp_path / "quality"
    quality.mkdir()
    pl.DataFrame(
        {
            **base,
            "logit": [0.0] * 4,
            "severity": [0.0] * 4,
            "reject_prob": [0.1, 0.9, 0.2, 0.8],
        },
        schema_overrides={"quality": pl.Int8},
    ).write_parquet(quality / "predictions.parquet")
    out = tmp_path / "ens"
    code = main(
        [
            "ensemble",
            "--run-dirs",
            str(dr),
            "--quality-run",
            str(quality),
            "--out-dir",
            str(out),
            "--resamples",
            "0",
        ]
    )
    assert code == 0
    predictions = pl.read_parquet(out / "predictions.parquet")
    assert predictions["reject_prob"].to_list() == [0.1, 0.9, 0.2, 0.8]
    assert predictions["logit"].to_list() == [-2.0, 2.0, -2.0, 2.0]
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["quality_run"] == str(quality)
    assert metrics["quality_threshold"] is not None
