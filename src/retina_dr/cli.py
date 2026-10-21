"""Command line: `retina-dr download` then `retina-dr build`."""

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import polars as pl

from retina_dr.download import download_sources
from retina_dr.labels import DEFAULT_BINARY_CUTOFF
from retina_dr.pipeline import BuildConfig, SourceReadError, build, summarize
from retina_dr.preprocess import DEFAULT_SIZE
from retina_dr.quality import attach_quality, read_eyeq
from retina_dr.records import SOURCES, Source
from retina_dr.split import SplitConfig

_BY_NAME: Final[dict[str, Source]] = {s: s for s in SOURCES}


def parse_sources(value: str) -> tuple[Source, ...]:
    """Parse a comma-separated source list, keeping its order."""
    names = [n.strip() for n in value.split(",") if n.strip()]
    unknown = [n for n in names if n not in _BY_NAME]
    if unknown:
        raise SystemExit(f"unknown source(s) {unknown}; choose from {list(SOURCES)}")
    return tuple(_BY_NAME[n] for n in names)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retina-dr", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    dl = commands.add_parser("download", help="download and extract sources from Kaggle")
    dl.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    dl.add_argument("--sources", default=",".join(SOURCES))
    dl.add_argument("--keep-archives", action="store_true", help="keep .zip files after extract")

    bd = commands.add_parser("build", help="preprocess images and write manifest.parquet")
    bd.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    bd.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    bd.add_argument("--sources", default=",".join(SOURCES))
    bd.add_argument("--size", type=int, default=DEFAULT_SIZE)
    bd.add_argument("--binary-cutoff", type=int, default=DEFAULT_BINARY_CUTOFF)
    bd.add_argument("--holdout", default="ddr", help="source to hold out, or 'none'")
    bd.add_argument("--seed", type=int, default=0)
    bd.add_argument("--workers", type=int, default=os.cpu_count() or 4)

    tr = commands.add_parser("train", help="train the two-head model and evaluate it")
    tr.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    tr.add_argument("--run-dir", type=Path, required=True)
    tr.add_argument("--model", default="efficientnet_b3")
    tr.add_argument("--no-pretrained", action="store_true")
    tr.add_argument("--image-size", type=int, default=512)
    tr.add_argument("--epochs", type=int, default=15)
    tr.add_argument("--batch-size", type=int, default=24)
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--weight-decay", type=float, default=1e-5)
    tr.add_argument("--layer-decay", type=float, default=None, help="ViT layer-wise lr decay")
    tr.add_argument("--severity-weight", type=float, default=1.0)
    tr.add_argument("--quality-weight", type=float, default=1.0)
    tr.add_argument("--task", choices=("dr", "quality"), default="dr")
    tr.add_argument(
        "--no-quality-labels", action="store_true", help="train without EyeQ labels (D-039)"
    )
    tr.add_argument("--rebalance-start", type=float, default=0.5)
    tr.add_argument("--workers", type=int, default=None, help="default: available CPUs - 1")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--limit", type=int, default=None, help="rows per split, for smoke runs")
    tr.add_argument("--no-tta", action="store_true")
    tr.add_argument("--log-every", type=int, default=200)
    tr.add_argument("--device", default="auto")
    aq = commands.add_parser("add-quality", help="add EyeQ quality labels to the manifest")
    aq.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    aq.add_argument("--eyeq-csv", type=Path, required=True)

    en = commands.add_parser("ensemble", help="average finished runs and evaluate (D-026)")
    en.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    en.add_argument("--out-dir", type=Path, required=True)
    en.add_argument("--quality-run", type=Path, default=None, help="run that gives reject_prob")
    en.add_argument("--resamples", type=int, default=500)

    bx = commands.add_parser("build-external", help="process the external test sets (D-035)")
    bx.add_argument("--raw-dir", type=Path, required=True, help="holds idrid/, deepdrid/, ...")
    bx.add_argument("--out-dir", type=Path, required=True)
    bx.add_argument("--size", type=int, default=DEFAULT_SIZE)

    sx = commands.add_parser("score-external", help="score the frozen model once (D-029)")
    sx.add_argument("--data-dir", type=Path, required=True)
    sx.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    sx.add_argument("--quality-run", type=Path, required=True)
    sx.add_argument("--ensemble-run", type=Path, required=True)
    sx.add_argument("--out-dir", type=Path, required=True)
    sx.add_argument("--resamples", type=int, default=500)

    ev = commands.add_parser("evaluate", help="recompute metrics of a finished run (D-017)")
    ev.add_argument("--run-dir", type=Path, required=True)
    ev.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    ev.add_argument("--resamples", type=int, default=500)
    return parser


def _add_quality(args: argparse.Namespace) -> int:
    path = Path(args.data_dir) / "manifest.parquet"
    manifest = attach_quality(pl.read_parquet(path), read_eyeq(Path(args.eyeq_csv)))
    manifest.write_parquet(path)
    counts = manifest.group_by("quality").len().sort("quality")
    print(f"quality labels added to {path}: {dict(counts.iter_rows())}")
    return 0


def _ensemble(args: argparse.Namespace) -> int:
    from retina_dr.train.report import ensemble_runs

    runs = [Path(d) for d in args.run_dirs]
    quality = None if args.quality_run is None else Path(args.quality_run)
    threshold, rows = ensemble_runs(runs, Path(args.out_dir), int(args.resamples), quality)
    print(f"ensemble of {len(runs)} runs, threshold {threshold:.4f}")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=400):
        print(pl.DataFrame(rows))
    return 0


def _build_external(args: argparse.Namespace) -> int:
    from retina_dr.external import build_external

    manifest = build_external(Path(args.raw_dir), Path(args.out_dir), int(args.size))
    print(manifest.group_by("source").len().sort("source"))
    return 0


def _score_external(args: argparse.Namespace) -> int:
    from retina_dr.train.external_eval import score_external

    result = score_external(
        Path(args.data_dir),
        [Path(d) for d in args.run_dirs],
        Path(args.quality_run),
        Path(args.ensemble_run),
        Path(args.out_dir),
        n_resamples=int(args.resamples),
    )
    print(json.dumps(result, indent=2))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from retina_dr.train.report import OPERATING_SPECIFICITY, reevaluate_run

    threshold, rows = reevaluate_run(Path(args.run_dir), Path(args.data_dir), int(args.resamples))
    print(f"threshold {threshold:.4f} (val specificity {OPERATING_SPECIFICITY})")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=400):
        print(pl.DataFrame(rows))
    return 0


def _train(args: argparse.Namespace) -> int:
    # Imported here so the data commands do not load PyTorch.
    from retina_dr.train.engine import TrainConfig, available_cpus, train

    config = TrainConfig(
        data_dir=Path(args.data_dir),
        run_dir=Path(args.run_dir),
        model=str(args.model),
        pretrained=not args.no_pretrained,
        image_size=int(args.image_size),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        layer_decay=None if args.layer_decay is None else float(args.layer_decay),
        severity_weight=float(args.severity_weight),
        quality_weight=float(args.quality_weight),
        task=str(args.task),
        quality_labels=not args.no_quality_labels,
        rebalance_start=float(args.rebalance_start),
        workers=max(1, available_cpus() - 1) if args.workers is None else int(args.workers),
        seed=int(args.seed),
        limit=None if args.limit is None else int(args.limit),
        tta=not args.no_tta,
        log_every=int(args.log_every),
        device=str(args.device),
    )
    print(f"run written to {train(config)}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parser().parse_args(argv)
    if args.command == "train":
        return _train(args)
    if args.command == "add-quality":
        return _add_quality(args)
    if args.command == "build-external":
        return _build_external(args)
    if args.command == "score-external":
        return _score_external(args)
    if args.command == "ensemble":
        return _ensemble(args)
    if args.command == "evaluate":
        return _evaluate(args)
    sources = parse_sources(str(args.sources))

    if args.command == "download":
        download_sources(Path(args.raw_dir), sources, delete_archives=not args.keep_archives)
        return 0

    holdout = None if args.holdout == "none" else parse_sources(str(args.holdout))[0]
    config = BuildConfig(
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
        sources=sources,
        size=int(args.size),
        binary_cutoff=int(args.binary_cutoff),
        split=SplitConfig(holdout_source=holdout, seed=int(args.seed)),
        workers=int(args.workers),
    )
    code = 0
    try:
        manifest = build(config)
    except SourceReadError as error:
        print(f"ERROR: {error}")
        manifest = pl.read_parquet(config.out_dir / "manifest.parquet")
        code = 2
    with pl.Config(tbl_rows=-1):
        print(summarize(manifest))
    print(f"{manifest.height} images -> {config.out_dir / 'manifest.parquet'}")
    return code


if __name__ == "__main__":
    sys.exit(main())
