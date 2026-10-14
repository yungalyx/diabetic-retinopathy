"""Adapter for the untyped kaggle package. All kaggle calls go through this module."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


class KaggleAccessError(RuntimeError):
    """Kaggle refused a download, usually because credentials or competition rules are missing."""


@dataclass(frozen=True, slots=True)
class KaggleRef:
    kind: Literal["competition", "dataset"]
    slug: str

    @property
    def url(self) -> str:
        base = "c" if self.kind == "competition" else "datasets"
        return f"https://www.kaggle.com/{base}/{self.slug}"


def download(ref: KaggleRef, dest: Path) -> None:
    """Download all files of a Kaggle competition or dataset into dest as archives."""
    # Imported here: the kaggle package reads credentials at import time.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    dest.mkdir(parents=True, exist_ok=True)
    try:
        if ref.kind == "competition":
            api.competition_download_files(ref.slug, path=str(dest), quiet=False)
        else:
            api.dataset_download_files(ref.slug, path=str(dest), quiet=False, unzip=False)
    except Exception as error:  # kaggle raises untyped HTTP errors; wrap them once here.
        hint = "Accept the competition rules at " if ref.kind == "competition" else "Check "
        raise KaggleAccessError(f"{ref.slug}: {error}. {hint}{ref.url}") from error
