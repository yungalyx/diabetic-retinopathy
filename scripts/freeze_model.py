"""Write runs/exp/<ensemble>/frozen_model.json: the record of the frozen model (D-035).

Run on a pod with the volume at /workspace:
    python scripts/freeze_model.py b3-e10-ens3-q
"""

import hashlib
import json
import sys
from pathlib import Path

RUNS = Path("/workspace/runs/exp")
MANIFEST = Path("/workspace/processed/manifest.parquet")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(name: str) -> None:
    ensemble = RUNS / name
    metrics = json.loads((ensemble / "metrics.json").read_text())
    members = [Path(m) for m in metrics["members"]]
    quality = Path(metrics["quality_run"])
    record = {
        "ensemble": str(ensemble),
        "ensemble_metrics_sha256": sha256(ensemble / "metrics.json"),
        "members": {
            str(m): {
                "commit": (m / "COMMIT").read_text().strip() if (m / "COMMIT").exists() else None,
                "best_pt_sha256": sha256(m / "best.pt"),
            }
            for m in [*members, quality]
        },
        "processed_manifest_sha256": sha256(MANIFEST),
        "processed_set_built_at": "2ca78e1 (before the D-022 crop fix)",
        "cutpoints": metrics["cutpoints"],
        "threshold": metrics["threshold"],
        "quality_threshold": metrics["quality_threshold"],
    }
    (ensemble / "frozen_model.json").write_text(json.dumps(record, indent=2))
    print("FROZEN", json.dumps(record))


if __name__ == "__main__":
    main(sys.argv[1])
