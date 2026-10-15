"""Download stage: fetch each source from Kaggle into raw_dir/<source> and extract it."""

import logging
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Final

from retina_dr.kaggle_adapter import KaggleRef, download
from retina_dr.records import Source
from retina_dr.sources import IMAGE_EXTENSIONS

log = logging.getLogger(__name__)

KAGGLE_SOURCES: Final[dict[Source, KaggleRef]] = {
    "eyepacs": KaggleRef("competition", "diabetic-retinopathy-detection"),
    "aptos": KaggleRef("competition", "aptos2019-blindness-detection"),
    "ddr": KaggleRef("dataset", "mariaherrerot/ddrdataset"),
    "odir": KaggleRef("dataset", "andrewmvd/ocular-disease-recognition-odir5k"),
}

PEAK_DISK_GB: Final[dict[Source, int]] = {
    "eyepacs": 180,
    "aptos": 20,
    "ddr": 30,
    "odir": 5,
}
"""Peak free disk per source when archives are deleted after extraction."""

KEEP_ARCHIVES_FACTOR: Final = 1.5
"""Extra disk factor when archives are kept (EyePACS then holds three copies)."""

_GIB: Final = 2**30
_MAX_PASSES: Final = 4


class InsufficientDiskError(RuntimeError):
    """Not enough free disk for the requested sources."""


def check_free_space(
    dest: Path,
    sources: tuple[Source, ...],
    free_bytes: int | None = None,
    delete_archives: bool = True,
) -> None:
    """Raise when free disk at dest is below the peak the sources need."""
    if free_bytes is None:
        dest.mkdir(parents=True, exist_ok=True)
        free_bytes = shutil.disk_usage(dest).free
    factor = 1.0 if delete_archives else KEEP_ARCHIVES_FACTOR
    needed_gb = round(sum(PEAK_DISK_GB[s] for s in sources) * factor)
    free_gb = free_bytes / _GIB
    if free_gb < needed_gb:
        raise InsufficientDiskError(
            f"{', '.join(sources)} need about {needed_gb} GB free; {free_gb:.0f} GB available. "
            "Use a larger volume or download fewer sources."
        )


def find_archives(root: Path) -> tuple[list[Path], list[Path]]:
    """Return (plain .zip files, first parts of multi-part archives) under root."""
    files = sorted(p for p in root.rglob("*") if p.is_file())
    plain = [p for p in files if p.suffix.lower() == ".zip"]
    multipart = [p for p in files if p.name.endswith(".001")]
    return plain, multipart


def _parts(head: Path) -> list[Path]:
    stem = head.name.removesuffix(".001")
    return sorted(head.parent.glob(f"{stem}.[0-9][0-9][0-9]"))


def _extract_multipart(head: Path) -> None:
    if shutil.which("7z") is None:
        raise RuntimeError(
            "7z is required for multi-part archives. Run: apt-get install -y p7zip-full"
        )
    subprocess.run(["7z", "x", "-y", f"-o{head.parent}", str(head)], check=True)


def extract_archives(root: Path, delete: bool) -> None:
    """Extract every archive under root, including archives inside archives."""
    done: set[Path] = set()
    for _ in range(_MAX_PASSES):
        plain, multipart = find_archives(root)
        pending = [p for p in [*plain, *multipart] if p not in done]
        if not pending:
            return
        for archive in pending:
            log.info("extracting %s", archive.name)
            if archive in multipart:
                _extract_multipart(archive)
                parts = _parts(archive)
            else:
                with zipfile.ZipFile(archive) as z:
                    z.extractall(archive.parent)
                parts = [archive]
            done.update(parts)
            if delete:
                for part in parts:
                    part.unlink()
    log.warning("archives still nested after %d passes under %s", _MAX_PASSES, root)


def is_extracted(dest: Path) -> bool:
    """Return True when dest holds images and no archives, so a rerun can skip it."""
    if not dest.is_dir():
        return False
    plain, multipart = find_archives(dest)
    if plain or multipart:
        return False
    return any(p.suffix.lower() in IMAGE_EXTENSIONS for p in dest.rglob("*"))


def download_sources(raw_dir: Path, sources: tuple[Source, ...], delete_archives: bool) -> None:
    """Download and extract each source into raw_dir/<source>."""
    pending = tuple(s for s in sources if not is_extracted(raw_dir / s))
    for source in sorted(set(sources) - set(pending)):
        log.info("%s: already extracted in %s, skipping download", source, raw_dir / source)
    if not pending:
        return
    check_free_space(raw_dir, pending, delete_archives=delete_archives)
    for source in pending:
        dest = raw_dir / source
        log.info("downloading %s into %s", source, dest)
        download(KAGGLE_SOURCES[source], dest)
        extract_archives(dest, delete=delete_archives)
