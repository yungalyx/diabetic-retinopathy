"""Print experiment results from the network volume and check HuggingFace access.

Run on a CPU pod with the volume at /workspace and HF_TOKEN set from the hf_token secret.
"""

import json
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

RUNS = Path("/workspace/runs/exp")
REPOS = ("YukunZhou/RETFound_mae_natureCFP", "YukunZhou/RETFound_dinov2_meh")
KEYS = (
    "split",
    "source",
    "auc",
    "sens_op",
    "spec_op",
    "sens_op_lo",
    "qwk_tuned",
    "qwk_tuned_lo",
    "sens_vtdr_op",
    "severe_undergraded",
    "severe_undergraded_rate",
    "sens_at_spec90",
)


def print_runs() -> None:
    for path in sorted(RUNS.glob("*/metrics.json")):
        name = path.parent.name
        metrics = json.loads(path.read_text())
        print("RUN", name, "best_epoch", metrics.get("best_epoch"), metrics.get("threshold"))
        for row in metrics["rows"]:
            if row["split"] in ("test", "holdout") and row["source"] in ("all", "ddr"):
                print("ROW", name, {k: row.get(k) for k in KEYS})
        history = path.parent / "history.jsonl"
        if history.exists():
            for line in history.read_text().splitlines():
                d = json.loads(line)
                print("HIST", name, d["epoch"], round(d["val_auc"], 4), round(d["val_qwk"], 4))


def check_hf() -> None:
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    try:
        print("WHOAMI", api.whoami()["name"])
    except Exception as error:
        print("WHOAMI_ERR", type(error).__name__, str(error)[:200])
    for repo in REPOS:
        try:
            files = [(f.path, getattr(f, "size", None)) for f in api.list_repo_tree(repo)]
            print("FILES", repo, files)
            weights = [p for p, _ in files if p.endswith((".pth", ".bin", ".safetensors"))]
            local = hf_hub_download(repo, weights[0], token=token, cache_dir="/workspace/.hf-cache")
            print("DOWNLOAD_OK", repo, weights[0], Path(local).stat().st_size)
        except Exception as error:
            print("ACCESS_ERR", repo, type(error).__name__, str(error)[:300])


if __name__ == "__main__":
    print_runs()
    check_hf()
    print("PROBE_DONE")
