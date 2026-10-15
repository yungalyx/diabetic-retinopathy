import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from retina_dr.train.data import FundusDataset, eval_transform, train_transform
from retina_dr.train.engine import TrainConfig, dihedral_views, train
from retina_dr.train.losses import two_head_loss
from retina_dr.train.model import build_model
from retina_dr.train.report import evaluate_predictions
from tests.conftest import write_fundus

TINY_MODEL = "resnet10t"


def _processed_dir(root: Path) -> Path:
    """Write a tiny processed set: images plus a manifest with all four splits."""
    rows = []
    layout = [("train", 16), ("val", 8), ("test", 4), ("holdout", 4)]
    index = 0
    for split, count in layout:
        for _ in range(count):
            grade = index % 5
            source = "odir" if split == "train" and index % 4 == 0 else "eyepacs"
            path = f"images/{source}/img{index}.jpg"
            write_fundus(root / path, seed=index, size=(48, 48))
            rows.append(
                {
                    "path": path,
                    "source": source,
                    "image_id": f"img{index}",
                    "patient_id": str(index),
                    "grade": None if source == "odir" else grade,
                    "binary_label": int(grade >= 2),
                    "split": split,
                    "dup_group": index,
                }
            )
            index += 1
    pl.DataFrame(rows, schema_overrides={"grade": pl.Int8, "binary_label": pl.Int8}).write_parquet(
        root / "manifest.parquet"
    )
    return root


def test_severity_loss_ignores_rows_without_a_grade() -> None:
    logit, severity = torch.zeros(3), torch.tensor([1.0, 2.0, 3.0])
    binary, has_grade = torch.tensor([0.0, 1.0, 1.0]), torch.tensor([1.0, 1.0, 0.0])
    a = two_head_loss(logit, severity, binary, torch.tensor([1.0, 2.0, 0.0]), has_grade)
    b = two_head_loss(logit, severity, binary, torch.tensor([1.0, 2.0, 4.0]), has_grade)
    assert a.total.item() == pytest.approx(b.total.item())
    assert a.severity.item() == pytest.approx(0.0)


def test_severity_loss_is_zero_when_no_row_has_a_grade() -> None:
    parts = two_head_loss(
        torch.zeros(2), torch.ones(2), torch.zeros(2), torch.full((2,), 4.0), torch.zeros(2)
    )
    assert parts.severity.item() == 0.0


def test_model_returns_one_logit_and_one_severity_per_image() -> None:
    model = build_model(TINY_MODEL, pretrained=False).eval()
    logit, severity = model(torch.zeros(2, 3, 64, 64))
    assert logit.shape == severity.shape == (2,)


def test_dihedral_views_are_eight_distinct_views_including_the_original() -> None:
    images = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    views = list(dihedral_views(images))
    assert len(views) == 8
    assert torch.equal(views[0], images)
    assert len({tuple(v.flatten().tolist()) for v in views}) == 8


def test_dataset_marks_rows_without_grade(tmp_path: Path) -> None:
    root = _processed_dir(tmp_path)
    frame = pl.read_parquet(root / "manifest.parquet").filter(pl.col("split") == "train")
    dataset = FundusDataset(frame, root, eval_transform(32))
    image, _, grade, has_grade = dataset[0]
    assert image.shape == (3, 32, 32)
    assert has_grade.item() == 0.0
    assert grade.item() == 0.0
    image, _, _, has_grade = FundusDataset(frame, root, train_transform(32))[1]
    assert image.shape == (3, 32, 32)
    assert has_grade.item() == 1.0


def test_report_gives_perfect_scores_for_perfect_predictions() -> None:
    frame = pl.DataFrame(
        {
            "split": ["test"] * 6,
            "source": ["a", "a", "a", "b", "b", "b"],
            "binary_label": [0, 0, 1, 0, 1, 1],
            "grade": [0, 1, 2, 0, 3, 4],
            "prob": [0.1, 0.2, 0.9, 0.1, 0.8, 0.95],
            "severity": [0.0, 1.0, 2.0, 0.0, 3.0, 4.0],
        }
    )
    rows = evaluate_predictions(frame, (0.5, 1.5, 2.5, 3.5))
    overall = next(r for r in rows if r["source"] == "all")
    assert overall["auc"] == 1.0
    assert overall["qwk_tuned"] == 1.0
    assert {r["source"] for r in rows} == {"all", "a", "b"}


def test_training_writes_checkpoint_history_predictions_and_metrics(tmp_path: Path) -> None:
    data = _processed_dir(tmp_path / "data")
    config = TrainConfig(
        data_dir=data,
        run_dir=tmp_path / "run",
        model=TINY_MODEL,
        pretrained=False,
        image_size=32,
        epochs=2,
        batch_size=4,
        workers=0,
        log_every=1,
        device="cpu",
    )
    run = train(config)
    history = (run / "history.jsonl").read_text().splitlines()
    assert len(history) == 2
    assert (run / "best.pt").is_file()
    predictions = pl.read_parquet(run / "predictions.parquet")
    assert set(predictions["split"].unique()) == {"val", "test", "holdout"}
    assert predictions.height == 16
    assert np.isfinite(predictions["prob"].to_numpy()).all()
    metrics = json.loads((run / "metrics.json").read_text())
    assert len(metrics["cutpoints"]) == 4
    assert any(r["split"] == "holdout" for r in metrics["rows"])
