import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from retina_dr.train.data import FundusDataset, eval_transform, train_transform
from retina_dr.train.engine import TrainConfig, dihedral_views, train
from retina_dr.train.losses import LossParts, model_loss
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
                    "quality": None if source == "odir" else index % 3,
                }
            )
            index += 1
    pl.DataFrame(
        rows, schema_overrides={"grade": pl.Int8, "binary_label": pl.Int8, "quality": pl.Int8}
    ).write_parquet(root / "manifest.parquet")
    return root


def _loss(
    binary: list[float],
    grade: list[float],
    has_grade: list[float],
    reject: list[float] | None = None,
    has_quality: list[float] | None = None,
    severity: list[float] | None = None,
) -> LossParts:
    n = len(binary)
    return model_loss(
        torch.zeros(n),
        torch.tensor(severity if severity is not None else [1.0] * n),
        torch.zeros(n),
        torch.tensor(binary),
        torch.tensor(grade),
        torch.tensor(has_grade),
        torch.tensor(reject if reject is not None else [0.0] * n),
        torch.tensor(has_quality if has_quality is not None else [0.0] * n),
    )


def test_severity_loss_ignores_rows_without_a_grade() -> None:
    a = _loss([0.0, 1.0, 1.0], [1.0, 2.0, 0.0], [1.0, 1.0, 0.0], severity=[1.0, 2.0, 3.0])
    b = _loss([0.0, 1.0, 1.0], [1.0, 2.0, 4.0], [1.0, 1.0, 0.0], severity=[1.0, 2.0, 3.0])
    assert a.total.item() == pytest.approx(b.total.item())
    assert a.severity.item() == pytest.approx(0.0)


def test_severity_loss_is_zero_when_no_row_has_a_grade() -> None:
    assert _loss([0.0, 0.0], [4.0, 4.0], [0.0, 0.0]).severity.item() == 0.0


def test_rejected_images_add_nothing_to_the_grading_losses() -> None:
    quality = {"reject": [0.0, 1.0], "has_quality": [1.0, 1.0]}
    a = _loss([0.0, 0.0], [1.0, 0.0], [1.0, 1.0], **quality)
    b = _loss([0.0, 1.0], [1.0, 4.0], [1.0, 1.0], **quality)
    assert a.binary.item() == pytest.approx(b.binary.item())
    assert a.severity.item() == pytest.approx(b.severity.item())
    assert a.quality.item() > 0.0


def test_quality_loss_is_zero_without_quality_labels() -> None:
    assert _loss([0.0, 1.0], [0.0, 2.0], [1.0, 1.0], reject=[1.0, 1.0]).quality.item() == 0.0


def test_model_returns_a_logit_a_severity_and_a_quality_logit_per_image() -> None:
    model = build_model(TINY_MODEL, pretrained=False).eval()
    logit, severity, quality = model(torch.zeros(2, 3, 64, 64))
    assert logit.shape == severity.shape == quality.shape == (2,)


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
    image, _, grade, has_grade, _, has_quality = dataset[0]
    assert image.shape == (3, 32, 32)
    assert has_grade.item() == 0.0
    assert grade.item() == 0.0
    assert has_quality.item() == 0.0
    image, _, _, has_grade, _, _ = FundusDataset(frame, root, train_transform(32))[1]
    assert image.shape == (3, 32, 32)
    assert has_grade.item() == 1.0


def test_dataset_reads_quality_labels_when_the_manifest_has_them(tmp_path: Path) -> None:
    root = _processed_dir(tmp_path)
    frame = pl.read_parquet(root / "manifest.parquet").head(3)
    frame = frame.with_columns(pl.Series("quality", [2, None, 0], dtype=pl.Int8))
    dataset = FundusDataset(frame, root, eval_transform(32))
    assert [dataset[i][4].item() for i in range(3)] == [1.0, 0.0, 0.0]
    assert [dataset[i][5].item() for i in range(3)] == [1.0, 0.0, 1.0]


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
    rows = evaluate_predictions(frame, (0.5, 1.5, 2.5, 3.5), threshold=0.5, n_resamples=20)
    overall = next(r for r in rows if r["source"] == "all")
    assert overall["auc"] == 1.0
    assert overall["qwk_tuned"] == 1.0
    assert overall["sens_op"] == overall["spec_op"] == 1.0
    assert overall["sens_vtdr_op"] == 1.0
    assert overall["severe_undergraded"] == 0
    assert overall["auc_lo"] == overall["auc_hi"] == 1.0
    assert {r["source"] for r in rows} == {"all", "a", "b"}


def test_report_counts_severe_undergrading_and_missed_vision_threatening_eyes() -> None:
    frame = pl.DataFrame(
        {
            "split": ["test"] * 4,
            "source": ["a"] * 4,
            "patient_id": ["p1", "p1", "p2", "p3"],
            "binary_label": [0, 1, 1, 1],
            "grade": [0, 3, 4, 2],
            "prob": [0.1, 0.2, 0.9, 0.8],
            "severity": [0.0, 1.0, 3.6, 2.0],
        }
    )
    row = evaluate_predictions(frame, (0.5, 1.5, 2.5, 3.5), threshold=0.5, n_resamples=0)[0]
    assert row["vtdr_images"] == 2
    assert row["sens_vtdr_op"] == 0.5
    assert row["severe_undergraded"] == 1
    assert row["severe_undergraded_rate"] == 0.5
    assert math.isnan(float(row["auc_lo"]))


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
    assert 0.0 <= metrics["threshold"] <= 1.0
    assert "patient_id" in predictions.columns
    saved = json.loads((run / "config.json").read_text())
    from retina_dr.provenance import file_sha256

    assert saved["provenance"]["manifest_sha256"] == file_sha256(data / "manifest.parquet")
    assert saved["provenance"]["weights_licence"]
    assert predictions["reject_prob"].is_between(0.0, 1.0).all()
    assert 0.0 <= metrics["quality_threshold"] <= 1.0
    holdout = next(r for r in metrics["rows"] if r["split"] == "holdout")
    assert 0.0 <= holdout["yield"] <= 1.0
    assert any(r["split"] == "holdout" for r in metrics["rows"])


def test_transforms_take_a_pil_image_and_return_a_normalised_tensor() -> None:
    from PIL import Image

    image = Image.new("RGB", (64, 64), (120, 60, 30))
    for transform in (train_transform(32), eval_transform(32)):
        out = transform(image)
        assert out.shape == (3, 32, 32)
        assert out.dtype == torch.float32


@pytest.mark.parametrize(
    ("cpu_max", "expected"),
    [("800000 100000\n", 8), ("150000 100000\n", 2), ("max 100000\n", 6), (None, 6)],
)
def test_available_cpus_reads_the_cgroup_quota(
    tmp_path: Path, cpu_max: str | None, expected: int
) -> None:
    from retina_dr.train.engine import available_cpus

    path = tmp_path / "cpu.max"
    if cpu_max is not None:
        path.write_text(cpu_max)
    assert available_cpus(path, fallback=6) == expected


def test_backbone_normalization_follows_the_pretrained_weights() -> None:
    from retina_dr.train.backbone import create_backbone, normalization_of
    from retina_dr.train.data import IMAGENET_MEAN

    tiny, _ = create_backbone(TINY_MODEL, pretrained=False)
    assert normalization_of(tiny)[0] == IMAGENET_MEAN
    vit, _ = create_backbone("vit_tiny_patch16_224.augreg_in21k", pretrained=False)
    assert normalization_of(vit) == ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def test_eval_transform_uses_the_given_normalization() -> None:
    from PIL import Image

    image = Image.new("RGB", (8, 8), (255, 255, 255))
    out = eval_transform(8, ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))(image)
    assert torch.allclose(out, torch.ones_like(out))


def _quality_frame(reject_prob: Sequence[float], quality: Sequence[int | None]) -> pl.DataFrame:
    n = len(reject_prob)
    return pl.DataFrame(
        {
            "split": ["val"] * n,
            "source": ["eyepacs"] * n,
            "binary_label": [i % 2 for i in range(n)],
            "grade": [2 * (i % 2) for i in range(n)],
            "quality": quality,
            "prob": [0.9 * (i % 2) for i in range(n)],
            "severity": [2.0 * (i % 2) for i in range(n)],
            "reject_prob": reject_prob,
        },
        schema_overrides={"quality": pl.Int8},
    )


def test_quality_threshold_keeps_at_least_ninety_percent_of_val_images() -> None:
    from retina_dr.train.report import quality_threshold

    # Half of the labelled images are reject: Youden alone would reject half of val.
    probs = [0.1 * i for i in range(10)]
    frame = _quality_frame(probs, [0] * 5 + [2] * 5)
    threshold = quality_threshold(frame)
    assert (np.array(probs) > threshold).mean() <= 0.10


def test_quality_threshold_uses_youden_when_the_yield_is_already_high() -> None:
    from retina_dr.train.report import quality_threshold

    probs = [0.0] * 18 + [0.95, 0.99]
    frame = _quality_frame(probs, [0] * 18 + [2, 2])
    threshold = quality_threshold(frame)
    assert 0.0 <= threshold < 0.95


def test_report_rejects_only_images_that_would_be_called_not_referable() -> None:
    # Rule B (D-032): a referred image is never rejected, whatever its quality.
    frame = pl.DataFrame(
        {
            "split": ["test"] * 4,
            "source": ["eyepacs"] * 4,
            "binary_label": [1, 1, 0, 0],
            "grade": [4, 3, 0, 0],
            "quality": [2, 2, 2, 0],
            "prob": [0.9, 0.1, 0.1, 0.1],
            "severity": [4.0, 0.0, 0.0, 0.0],
            "reject_prob": [0.9, 0.9, 0.9, 0.1],
        },
        schema_overrides={"quality": pl.Int8},
    )
    row = evaluate_predictions(
        frame, (0.5, 1.5, 2.5, 3.5), threshold=0.5, quality_threshold=0.5, n_resamples=0
    )[0]
    assert row["yield"] == 0.5
    assert row["reject_rate_pos"] == 0.5
    assert row["reject_rate_neg"] == 0.5
    assert row["reject_recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert row["accepted_images"] == 2
    assert row["sens_op_accepted"] == 1.0


def test_loss_without_binary_weight_ignores_the_binary_term() -> None:
    parts = model_loss(
        torch.zeros(2),
        torch.ones(2),
        torch.zeros(2),
        torch.tensor([0.0, 1.0]),
        torch.tensor([0.0, 2.0]),
        torch.zeros(2),
        torch.tensor([0.0, 1.0]),
        torch.ones(2),
        severity_weight=0.0,
        binary_weight=0.0,
    )
    assert parts.total.item() == pytest.approx(parts.quality.item())


def test_quality_task_trains_only_on_rows_with_a_quality_label(tmp_path: Path) -> None:
    data = _processed_dir(tmp_path / "data")
    config = TrainConfig(
        data_dir=data,
        run_dir=tmp_path / "run",
        model=TINY_MODEL,
        pretrained=False,
        image_size=32,
        epochs=1,
        batch_size=4,
        workers=0,
        log_every=1,
        device="cpu",
        task="quality",
        tta=False,
    )
    run = train(config)
    record = json.loads((run / "history.jsonl").read_text().splitlines()[0])
    assert record["train_rows"] == 12
    assert record["val_score"] == record["val_quality_auc"]


def test_system_grade_follows_the_referral_decision() -> None:
    from retina_dr.train.report import system_grades

    prob = np.array([0.9, 0.9, 0.1, 0.1])
    pred = np.array([0, 3, 4, 1])
    assert system_grades(prob, pred, threshold=0.5).tolist() == [2, 3, 1, 1]


def test_report_counts_head_disagreement_and_system_kappa() -> None:
    frame = pl.DataFrame(
        {
            "split": ["test"] * 4,
            "source": ["a"] * 4,
            "binary_label": [1, 1, 0, 0],
            "grade": [2, 4, 0, 1],
            "prob": [0.9, 0.1, 0.1, 0.1],
            "severity": [0.0, 4.0, 0.0, 1.0],
        }
    )
    row = evaluate_predictions(frame, (0.5, 1.5, 2.5, 3.5), threshold=0.5, n_resamples=0)[0]
    assert row["head_disagreement"] == 0.5
    assert row["severe_undergraded_system"] == 1
    assert float(row["qwk_system"]) < float(row["qwk_tuned"])


def test_eye_level_averages_the_images_of_one_eye() -> None:
    from retina_dr.train.report import to_eye_level

    frame = pl.DataFrame(
        {
            "path": ["a1", "a2", "b1"],
            "eye_id": ["a", "a", "b"],
            "split": ["test"] * 3,
            "source": ["s"] * 3,
            "patient_id": ["p", "p", "q"],
            "binary_label": [0, 1, 0],
            "grade": [1, 3, 0],
            "quality": [2, 0, 2],
            "prob": [0.2, 0.6, 0.1],
            "severity": [2.0, 4.0, 0.0],
            "reject_prob": [0.1, 0.3, 0.0],
        }
    )
    eyes = to_eye_level(frame).sort("eye_id")
    assert eyes.height == 2
    assert eyes["prob"].to_list() == pytest.approx([0.4, 0.1])
    assert eyes["severity"].to_list() == pytest.approx([3.0, 0.0])
    assert eyes["images"].to_list() == [2, 1]
    assert eyes["grade"].to_list() == [3, 0]
    assert eyes["binary_label"].to_list() == [1, 0]
    assert eyes["quality"].to_list() == [0, 2]


def test_eye_level_treats_each_image_as_one_eye_without_an_eye_id() -> None:
    from retina_dr.train.report import to_eye_level

    frame = pl.DataFrame(
        {
            "path": ["a", "b"],
            "split": ["test"] * 2,
            "source": ["s"] * 2,
            "binary_label": [1, 0],
            "grade": [3, 0],
            "prob": [0.2, 0.6],
            "severity": [2.0, 4.0],
        }
    )
    assert to_eye_level(frame).height == 2
