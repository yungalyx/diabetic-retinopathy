"""Training loop, checkpoint selection, test-time augmentation, and the final evaluation."""

import contextlib
import json
import logging
import math
import os
import random
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader, WeightedRandomSampler

from retina_dr.quality import REJECT
from retina_dr.train.backbone import normalization_of
from retina_dr.train.data import Batch, FundusDataset, eval_transform, labels_of, train_transform
from retina_dr.train.losses import model_loss
from retina_dr.train.metrics import (
    DEFAULT_CUTPOINTS,
    apply_cutpoints,
    quadratic_weighted_kappa,
    roc_auc,
    tune_cutpoints,
)
from retina_dr.train.model import DrNet, build_model
from retina_dr.train.report import (
    evaluate_predictions,
    operating_threshold,
    quality_threshold,
    sigmoid,
)
from retina_dr.train.sampling import rebalance_alpha, sample_weights, strata_from_labels

log = logging.getLogger(__name__)

TRAIN_SPLIT: Final = "train"
VAL_SPLIT: Final = "val"
EVAL_SPLITS: Final = ("val", "test", "holdout")
WARMUP_FRACTION: Final = 0.05

FloatArray = NDArray[np.float64]


CGROUP_CPU_MAX: Final = Path("/sys/fs/cgroup/cpu.max")


def available_cpus(cpu_max: Path = CGROUP_CPU_MAX, fallback: int | None = None) -> int:
    """CPUs this process may use: the cgroup v2 quota when set, else the affinity mask.

    os.cpu_count() reports the host's CPUs inside a container, which oversubscribes it.
    """
    default = fallback if fallback is not None else len(os.sched_getaffinity(0))
    try:
        quota, period = cpu_max.read_text().split()[:2]
    except (OSError, ValueError):
        return default
    if quota == "max":
        return default
    return max(1, math.ceil(int(quota) / int(period)))


def _worker_init(_: int) -> None:
    # Each loader worker decodes one image at a time; extra threads only add contention.
    torch.set_num_threads(1)


@dataclass(frozen=True, slots=True)
class TrainConfig:
    data_dir: Path
    run_dir: Path
    model: str = "efficientnet_b3"
    pretrained: bool = True
    image_size: int = 512
    epochs: int = 15
    batch_size: int = 24
    lr: float = 3e-4
    weight_decay: float = 1e-5
    severity_weight: float = 1.0
    quality_weight: float = 1.0
    rebalance_start: float = 0.5
    workers: int = 8
    seed: int = 0
    limit: int | None = None
    """Rows per split, for smoke runs. None uses every row."""
    tta: bool = True
    log_every: int = 200
    device: str = "auto"


@dataclass(frozen=True, slots=True)
class Predictions:
    logits: FloatArray
    severity: FloatArray
    quality: FloatArray
    """Logit of "reject" from the quality head (D-025)."""


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _autocast(device: torch.device) -> contextlib.AbstractContextManager[None]:
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def load_splits(config: TrainConfig) -> dict[str, pl.DataFrame]:
    manifest = pl.read_parquet(config.data_dir / "manifest.parquet")
    splits: dict[str, pl.DataFrame] = {}
    for split in (TRAIN_SPLIT, *EVAL_SPLITS):
        frame = manifest.filter(pl.col("split") == split)
        if config.limit is not None and frame.height > config.limit:
            frame = frame.sample(n=config.limit, seed=config.seed)
        splits[split] = frame
    return splits


def dihedral_views(images: Tensor) -> Iterator[Tensor]:
    """The 8 rotations and reflections of a square image batch (NCHW)."""
    for quarter_turns in range(4):
        rotated = torch.rot90(images, quarter_turns, dims=(2, 3))
        yield rotated
        yield rotated.flip(3)


def predict(
    model: DrNet,
    loader: DataLoader[Batch],
    device: torch.device,
    tta: bool,
) -> Predictions:
    """All three head outputs, averaged over dihedral views when tta is on."""
    model.eval()
    heads: tuple[list[FloatArray], list[FloatArray], list[FloatArray]] = ([], [], [])
    with torch.inference_mode(), _autocast(device):
        for batch_rows in loader:
            batch = batch_rows[0].to(device, non_blocking=True)
            views = list(dihedral_views(batch)) if tta else [batch]
            outputs = [model(view) for view in views]
            for index, store in enumerate(heads):
                mean = torch.stack([o[index].float() for o in outputs]).mean(0)
                store.append(mean.cpu().numpy().astype(np.float64))
    return Predictions(*(np.concatenate(store) for store in heads))


def _quality_auc(frame: pl.DataFrame, preds: Predictions) -> float:
    """Val AUC of the quality head on rows with a quality label (reject is positive)."""
    quality = labels_of(frame).quality
    labelled = quality >= 0
    if not labelled.any():
        return float("nan")
    return roc_auc((quality[labelled] == REJECT).astype(np.int64), preds.quality[labelled])


def _validation_score(
    frame: pl.DataFrame, preds: Predictions
) -> tuple[float, float, tuple[float, ...]]:
    """Val AUC, val QWK with cutpoints tuned on val, and those cutpoints."""
    labels = labels_of(frame)
    auc = roc_auc(labels.binary, preds.logits)
    graded = labels.grade >= 0
    if not graded.any():
        return auc, float("nan"), DEFAULT_CUTPOINTS
    cuts = tune_cutpoints(preds.severity[graded], labels.grade[graded])
    qwk = quadratic_weighted_kappa(
        labels.grade[graded], apply_cutpoints(preds.severity[graded], cuts)
    )
    return auc, qwk, cuts


def _lr_factor(step: int, total: int) -> float:
    warmup = max(1, int(total * WARMUP_FRACTION))
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _loader(
    dataset: FundusDataset,
    config: TrainConfig,
    device: torch.device,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader[Batch]:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=sampler is not None,
        num_workers=config.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.workers > 0,
        worker_init_fn=_worker_init if config.workers > 0 else None,
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))


def _with_quality_column(frame: pl.DataFrame) -> pl.DataFrame:
    if "quality" in frame.columns:
        return frame
    return frame.with_columns(pl.lit(None, dtype=pl.Int8).alias("quality"))


def train(config: TrainConfig) -> Path:
    """Train, keep the best epoch by (val AUC + val QWK) / 2, then evaluate it with TTA."""
    _seed_everything(config.seed)
    device = _device(config.device)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config.run_dir / "config.json", asdict(config))
    splits = load_splits(config)
    log.info("device %s, rows per split %s", device, {k: v.height for k, v in splits.items()})

    root = config.data_dir
    model = build_model(config.model, config.pretrained).to(device)
    norm = normalization_of(model.backbone)
    log.info("model %s, input normalization mean %s std %s", config.model, norm[0], norm[1])
    train_set = FundusDataset(splits[TRAIN_SPLIT], root, train_transform(config.image_size, norm))
    labels = labels_of(splits[TRAIN_SPLIT])
    strata = strata_from_labels(labels.grade, labels.binary)
    sampler = WeightedRandomSampler(
        [1.0] * len(train_set),
        num_samples=len(train_set),
        replacement=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    train_loader = _loader(train_set, config, device, sampler)
    eval_loaders = {
        split: _loader(
            FundusDataset(splits[split], root, eval_transform(config.image_size, norm)),
            config,
            device,
        )
        for split in EVAL_SPLITS
    }

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    total_steps = config.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _lr_factor(step, total_steps)
    )

    best_score = -math.inf
    history = (config.run_dir / "history.jsonl").open("w")
    for epoch in range(config.epochs):
        alpha = rebalance_alpha(epoch, config.epochs, config.rebalance_start)
        sampler.weights = torch.from_numpy(sample_weights(strata, alpha))
        model.train()
        started = time.perf_counter()
        sums = np.zeros(4)
        steps = 0
        for images, *targets in train_loader:
            batch = images.to(device, non_blocking=True)
            if device.type == "cuda":
                batch = batch.contiguous(memory_format=torch.channels_last)
            with _autocast(device):
                binary_logit, severity, quality_logit = model(batch)
            binary, grade, has_grade, reject, has_quality = (t.to(device) for t in targets)
            parts = model_loss(
                binary_logit,
                severity,
                quality_logit,
                binary,
                grade,
                has_grade,
                reject,
                has_quality,
                config.severity_weight,
                config.quality_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            torch.autograd.backward(parts.total)
            optimizer.step()
            scheduler.step()
            sums += [
                parts.total.item(),
                parts.binary.item(),
                parts.severity.item(),
                parts.quality.item(),
            ]
            steps += 1
            if steps % config.log_every == 0:
                rate = steps * config.batch_size / (time.perf_counter() - started)
                log.info(
                    "epoch %d step %d/%d loss %.4f %.0f img/s",
                    epoch,
                    steps,
                    len(train_loader),
                    sums[0] / steps,
                    rate,
                )
        preds = predict(model, eval_loaders[VAL_SPLIT], device, tta=False)
        auc, qwk, cuts = _validation_score(splits[VAL_SPLIT], preds)
        score = float(np.nanmean([auc, qwk]))
        record = {
            "epoch": epoch,
            "alpha": round(alpha, 4),
            "lr": scheduler.get_last_lr()[0],
            "train_loss": sums[0] / max(steps, 1),
            "train_bce": sums[1] / max(steps, 1),
            "train_huber": sums[2] / max(steps, 1),
            "train_quality": sums[3] / max(steps, 1),
            "val_auc": auc,
            "val_quality_auc": _quality_auc(splits[VAL_SPLIT], preds),
            "val_qwk": qwk,
            "val_score": score,
            "seconds": round(time.perf_counter() - started, 1),
        }
        history.write(json.dumps(record) + "\n")
        history.flush()
        log.info("epoch %d done: %s", epoch, json.dumps(record))
        if score > best_score:
            best_score = score
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "cuts": cuts, "val": record},
                config.run_dir / "best.pt",
            )
            log.info("epoch %d is the new best (score %.4f)", epoch, score)
    history.close()

    checkpoint = torch.load(config.run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    log.info("final evaluation with best epoch %d, tta=%s", checkpoint["epoch"], config.tta)
    frames: list[pl.DataFrame] = []
    for split in EVAL_SPLITS:
        if splits[split].height == 0:
            continue
        preds = predict(model, eval_loaders[split], device, tta=config.tta)
        frames.append(
            _with_quality_column(splits[split])
            .select(
                "path",
                "source",
                "image_id",
                "patient_id",
                "split",
                "binary_label",
                "grade",
                "quality",
            )
            .with_columns(
                pl.Series("logit", preds.logits),
                pl.Series("prob", sigmoid(preds.logits)),
                pl.Series("severity", preds.severity),
                pl.Series("reject_prob", sigmoid(preds.quality)),
            )
        )
    predictions = pl.concat(frames)
    val = predictions.filter((pl.col("split") == VAL_SPLIT) & pl.col("grade").is_not_null())
    cuts = (
        tune_cutpoints(val["severity"].to_numpy(), val["grade"].to_numpy())
        if val.height
        else DEFAULT_CUTPOINTS
    )
    predictions = predictions.with_columns(
        pl.Series("pred_grade", apply_cutpoints(predictions["severity"].to_numpy(), cuts))
    )
    predictions.write_parquet(config.run_dir / "predictions.parquet")
    threshold = operating_threshold(predictions)
    reject_threshold = quality_threshold(predictions)
    rows = evaluate_predictions(predictions, cuts, threshold, reject_threshold)
    _write_json(
        config.run_dir / "metrics.json",
        {
            "best_epoch": checkpoint["epoch"],
            "cutpoints": cuts,
            "threshold": threshold,
            "quality_threshold": reject_threshold,
            "rows": rows,
        },
    )
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        log.info("results (cutpoints tuned on val: %s)\n%s", cuts, pl.DataFrame(rows))
    return config.run_dir
