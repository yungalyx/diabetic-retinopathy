"""Score the frozen model on the external sets once (D-029, D-035).

Loads each member run's best.pt, predicts with test-time augmentation, averages the members,
takes reject_prob from the quality run, and applies the thresholds and cutpoints frozen in the
ensemble's metrics.json. Results are per eye (D-030, D-034).
"""

import json
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import torch

from retina_dr.provenance import file_sha256, git_commit
from retina_dr.train.backbone import normalization_of
from retina_dr.train.data import FundusDataset, eval_transform
from retina_dr.train.engine import TrainConfig, _device, _loader, predict
from retina_dr.train.metrics import apply_cutpoints
from retina_dr.train.model import DrNet, build_model
from retina_dr.train.report import (
    evaluate_predictions,
    rejected_rows,
    sigmoid,
    to_eye_level,
)

FloatArray = np.typing.NDArray[np.float64]


_OPTIONAL_KEYS: Final = frozenset({"quality_head.weight", "quality_head.bias"})
"""Runs trained before D-025 have no quality head. Their quality output is not used."""


def load_checkpoint(model: DrNet, path: Path, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    result = model.load_state_dict(checkpoint["model"], strict=False)
    missing = set(result.missing_keys) - _OPTIONAL_KEYS
    if missing or result.unexpected_keys:
        raise RuntimeError(
            f"{path}: missing {sorted(missing)}, unexpected {sorted(result.unexpected_keys)}"
        )


def _run_outputs(
    run_dir: Path, frame: pl.DataFrame, data_dir: Path, device: torch.device, tta: bool
) -> tuple[FloatArray, FloatArray, FloatArray]:
    saved = json.loads((run_dir / "config.json").read_text())
    model = build_model(str(saved["model"]), pretrained=False)
    load_checkpoint(model, run_dir / "best.pt", device)
    model = model.to(device)
    norm = normalization_of(model.backbone)
    size = int(saved["image_size"])
    config = TrainConfig(data_dir=data_dir, run_dir=run_dir, workers=0, batch_size=16)
    dataset = FundusDataset(frame, data_dir, eval_transform(size, norm))
    preds = predict(model, _loader(dataset, config, device), device, tta=tta)
    return preds.logits, preds.severity, preds.quality


def score_external(
    data_dir: Path,
    member_runs: list[Path],
    quality_run: Path,
    ensemble_run: Path,
    out_dir: Path,
    device: str = "auto",
    tta: bool = True,
    n_resamples: int = 500,
) -> dict[str, object]:
    manifest = pl.read_parquet(data_dir / "manifest.parquet")
    # Labels are not used for prediction; fill gaps so the dataset can load every image.
    frame = manifest.with_columns(pl.col("binary_label").fill_null(0), pl.col("grade").fill_null(0))
    torch_device = _device(device)
    outputs = [_run_outputs(r, frame, data_dir, torch_device, tta) for r in member_runs]
    logit = np.mean([o[0] for o in outputs], axis=0)
    severity = np.mean([o[1] for o in outputs], axis=0)
    quality = _run_outputs(quality_run, frame, data_dir, torch_device, tta)[2]

    frozen = json.loads((ensemble_run / "metrics.json").read_text())
    cuts = tuple(float(c) for c in frozen["cutpoints"])
    threshold = float(frozen["threshold"])
    reject_threshold = float(frozen["quality_threshold"])

    predictions = manifest.with_columns(
        pl.Series("logit", logit),
        pl.Series("prob", sigmoid(logit)),
        pl.Series("severity", severity),
        pl.Series("reject_prob", sigmoid(quality)),
        pl.Series("pred_grade", apply_cutpoints(severity, cuts)),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.write_parquet(out_dir / "predictions.parquet")

    eyes = to_eye_level(predictions)
    labelled = eyes.filter(pl.col("binary_label").is_not_null())
    rows = evaluate_predictions(
        labelled, cuts, threshold, reject_threshold, n_resamples=n_resamples
    )
    ungradable: dict[str, dict[str, float | int]] = {}
    for source in sorted(eyes["source"].unique().to_list()):
        part = eyes.filter((pl.col("source") == source) & pl.col("binary_label").is_null())
        if part.height:
            flagged = rejected_rows(part, threshold, reject_threshold)
            referred = part["prob"].to_numpy() > threshold
            ungradable[source] = {
                "eyes": part.height,
                "rejected": int(flagged.sum()),
                "referred": int(referred.sum()),
            }
    result: dict[str, object] = {
        "members": [str(r) for r in member_runs],
        "quality_run": str(quality_run),
        "ensemble_run": str(ensemble_run),
        "cutpoints": list(cuts),
        "threshold": threshold,
        "quality_threshold": reject_threshold,
        "unit": "eye",
        "rows": rows,
        "ungradable": ungradable,
        "provenance": {
            "commit": git_commit(Path(__file__).parent),
            "manifest_sha256": file_sha256(data_dir / "manifest.parquet"),
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    return result
