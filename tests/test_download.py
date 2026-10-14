import zipfile
from pathlib import Path

import pytest

from retina_dr.download import (
    InsufficientDiskError,
    check_free_space,
    extract_archives,
    find_archives,
)


def _zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path


def test_find_archives_separates_plain_zips_from_multipart_heads(tmp_path: Path) -> None:
    for name in ["a.zip", "train.zip.001", "train.zip.002", "notes.txt"]:
        (tmp_path / name).write_bytes(b"x")
    plain, multipart = find_archives(tmp_path)
    assert [p.name for p in plain] == ["a.zip"]
    assert [p.name for p in multipart] == ["train.zip.001"]


def test_extract_unpacks_nested_zips_and_deletes_archives(tmp_path: Path) -> None:
    inner = _zip(tmp_path / "trainLabels.csv.zip", {"trainLabels.csv": b"image,level\n"})
    _zip(tmp_path / "outer.zip", {"trainLabels.csv.zip": inner.read_bytes()})
    inner.unlink()
    extract_archives(tmp_path, delete=True)
    assert (tmp_path / "trainLabels.csv").read_text() == "image,level\n"
    assert not list(tmp_path.rglob("*.zip"))


def test_extract_keeps_archives_when_delete_is_off(tmp_path: Path) -> None:
    _zip(tmp_path / "a.zip", {"a.txt": b"1"})
    extract_archives(tmp_path, delete=False)
    assert (tmp_path / "a.zip").exists()
    assert (tmp_path / "a.txt").exists()


def test_free_space_check_raises_when_disk_is_too_small(tmp_path: Path) -> None:
    with pytest.raises(InsufficientDiskError, match="eyepacs"):
        check_free_space(tmp_path, ("eyepacs",), free_bytes=10 * 2**30)


def test_free_space_check_passes_with_enough_disk(tmp_path: Path) -> None:
    check_free_space(tmp_path, ("aptos", "odir"), free_bytes=500 * 2**30)
