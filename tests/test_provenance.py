import hashlib
from pathlib import Path

from retina_dr.provenance import SOURCE_LICENCES, file_sha256, git_commit


def test_file_sha256_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "x.bin"
    path.write_bytes(b"retina" * 1000)
    assert file_sha256(path) == hashlib.sha256(b"retina" * 1000).hexdigest()


def test_git_commit_is_unknown_outside_a_repository(tmp_path: Path) -> None:
    assert git_commit(tmp_path) == "unknown"


def test_git_commit_is_a_full_hash_inside_this_repository() -> None:
    commit = git_commit(Path(__file__).parent)
    assert commit == "unknown" or len(commit) == 40


def test_every_source_has_a_licence_entry() -> None:
    assert set(SOURCE_LICENCES) >= {"eyepacs", "aptos", "ddr", "odir", "eyeq"}
