import json, glob, os

for path in sorted(glob.glob("/workspace/runs/exp/*/metrics.json")):
    name = path.split("/")[-2]
    m = json.load(open(path))
    print("RUN", name, "best_epoch", m.get("best_epoch"), "threshold", m.get("threshold"))
    for r in m["rows"]:
        if r["split"] in ("test", "holdout") and r["source"] in ("all", "ddr"):
            keys = [
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
            ]
            print("ROW", name, {k: r.get(k) for k in keys})
    h = "/workspace/runs/exp/%s/history.jsonl" % name
    if os.path.exists(h):
        for line in open(h):
            d = json.loads(line)
            print(
                "HIST",
                name,
                d["epoch"],
                round(d["val_auc"], 4),
                round(d["val_qwk"], 4),
                d["seconds"],
            )
from huggingface_hub import HfApi

api = HfApi(token=os.environ.get("HF_TOKEN"))
try:
    print("WHOAMI", api.whoami()["name"])
except Exception as e:
    print("WHOAMI_ERR", type(e).__name__, str(e)[:200])
for repo in ["YukunZhou/RETFound_mae_natureCFP", "YukunZhou/RETFound_dinov2_meh"]:
    try:
        files = [
            (f.path, f.size) for f in api.list_repo_tree(repo, recursive=True) if hasattr(f, "size")
        ]
        print("FILES", repo, files)
        from huggingface_hub import hf_hub_download

        name = [p for p, _ in files if p.endswith((".pth", ".bin", ".safetensors"))][0]
        local = hf_hub_download(
            repo, name, token=os.environ.get("HF_TOKEN"), cache_dir="/workspace/.hf-cache"
        )
        print("DOWNLOAD_OK", repo, name, os.path.getsize(local))
    except Exception as e:
        print("ACCESS_ERR", repo, type(e).__name__, str(e)[:300])
print("PROBE_DONE")
