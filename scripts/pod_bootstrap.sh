#!/usr/bin/env bash
# Run at pod boot: get the code, download the data, and build the processed set.
# Raw data goes to the container disk (temporary). Processed data goes to the network volume.
set -euo pipefail

REPO_URL="https://github.com/yungalyx/diabetic-retinopathy"
WORK="/workspace"
CODE="$WORK/diabetic-retinopathy"
RAW="${RAW_DIR:-/root/raw}"
OUT="$WORK/processed"
STATUS="$WORK/status"

log() { echo "[bootstrap $(date -u +%H:%M:%S)] $*"; }

mkdir -p "$STATUS"
rm -f "$STATUS/DONE" "$STATUS/FAILED"
trap 'log "FAILED at line $LINENO"; echo "line $LINENO" > "$STATUS/FAILED"' ERR

if [[ -d "$CODE/.git" ]]; then
    log "updating code"
    git -C "$CODE" pull --ff-only
else
    log "cloning code"
    git clone "$REPO_URL" "$CODE"
fi
cd "$CODE"

log "running setup"
bash scripts/runpod_setup.sh
export PATH="$HOME/.local/bin:$PATH"

if [[ -f "$OUT/manifest.parquet" ]]; then
    log "manifest already exists at $OUT, skipping download and build"
else
    log "downloading to $RAW"
    uv run retina-dr download --raw-dir "$RAW"
    log "raw layout (folder, file count):"
    find "$RAW" -mindepth 1 -maxdepth 3 -type d | sort | while read -r dir; do
        echo "  $dir $(find "$dir" -maxdepth 1 -type f | wc -l)"
    done
    log "building to $OUT"
    uv run retina-dr build --raw-dir "$RAW" --out-dir "$OUT"
fi

du -sh "$OUT" || true
log "DONE"
touch "$STATUS/DONE"
