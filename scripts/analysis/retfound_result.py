"""Print the D-038 comparison rows from the network volume. Standard library only.

Run on a CPU pod with the volume at /workspace.
"""

import json
from pathlib import Path

RUNS = Path("/workspace/runs/exp")
NAMES = (
    "b3-e10-ens3-q",
    "retfound-e10",
    "retfound-e10-s1",
    "retfound-e10-s2",
    "retfound-ens3",
    "retfound-ens3-q",
)
KEYS = (
    "auc",
    "auc_lo",
    "auc_hi",
    "sens_op",
    "spec_op",
    "qwk_system",
    "qwk_tuned",
    "sens_vtdr_op",
    "severe_undergraded_system",
    "yield",
)

for name in NAMES:
    path = RUNS / name / "metrics.json"
    if not path.exists():
        print("MISSING", name)
        continue
    metrics = json.loads(path.read_text())
    print("RUN", name, "best_epoch", metrics.get("best_epoch"))
    for row in metrics["rows"]:
        if row["source"] == "all" and row["split"] in ("val", "test"):
            print("ROW", name, row["split"], {k: row.get(k) for k in KEYS})
print("DONE")
