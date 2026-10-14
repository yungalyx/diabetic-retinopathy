"""Command line: `retina-dr download` then `retina-dr build`."""

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import polars as pl

from retina_dr.download import download_sources
from retina_dr.labels import DEFAULT_BINARY_CUTOFF
from retina_dr.pipeline import BuildConfig, build, summarize
from retina_dr.preprocess import DEFAULT_SIZE
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parser().parse_args(argv)
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
    manifest = build(config)
    with pl.Config(tbl_rows=-1):
        print(summarize(manifest))
    print(f"{manifest.height} images -> {config.out_dir / 'manifest.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
