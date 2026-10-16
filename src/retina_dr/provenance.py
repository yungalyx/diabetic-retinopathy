"""Provenance records for builds and runs (D-019, D-031)."""

import hashlib
import subprocess
from pathlib import Path
from typing import Final

SOURCE_LICENCES: Final = {
    "eyepacs": "Kaggle competition rules (diabetic-retinopathy-detection)",
    "aptos": "Kaggle competition rules (aptos2019-blindness-detection)",
    "ddr": "CC BY-NC-SA 4.0 (evaluation only)",
    "odir": "not recorded (Kaggle mirror)",
    "eyeq": "CC BY-NC-SA 4.0 (quality labels)",
}
"""Licence or terms per data source. See docs/research/dataset-licences.md."""

_CHUNK: Final = 1 << 20


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(directory: Path) -> str:
    """Full commit hash of the repository that holds `directory`, or "unknown"."""
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    commit = result.stdout.strip()
    return commit if len(commit) == 40 else "unknown"


def run_provenance(manifest: Path, sources: list[str], weights_licence: str) -> dict[str, object]:
    """Commit, manifest hash, and licences for one training run."""
    return {
        "commit": git_commit(Path(__file__).parent),
        "manifest_sha256": file_sha256(manifest),
        "source_licences": {s: SOURCE_LICENCES.get(s, "not recorded") for s in sources},
        "weights_licence": weights_licence,
    }
