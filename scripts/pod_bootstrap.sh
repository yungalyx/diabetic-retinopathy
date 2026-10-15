#!/usr/bin/env bash
# Run at pod boot: get the code, download the data, and build the processed set.
# Raw data goes to the container disk (temporary). Processed data goes to the network volume.
# On failure, wait for a new commit on GitHub, pull it, and retry. Raw downloads are kept.
set -uo pipefail

REPO_URL="https://github.com/yungalyx/diabetic-retinopathy"
WORK="/workspace"
CODE="$WORK/diabetic-retinopathy"
RAW="${RAW_DIR:-/root/raw}"
OUT="$WORK/processed"
STATUS="$WORK/status"
POLL_SECONDS=60

log() { echo "[bootstrap $(date -u +%H:%M:%S)] $*"; }

show_raw() {
    log "raw layout (folder, file count):"
    find "$RAW" -mindepth 1 -maxdepth 4 -type d | sort | while read -r dir; do
        echo "  $dir $(find "$dir" -maxdepth 1 -type f | wc -l)"
    done
    log "label files (first 3 lines):"
    find "$RAW" -maxdepth 4 -type f \( -name "*.csv" -o -name "*.txt" \) | sort | while read -r file; do
        echo "  == $file"
        head -n 3 "$file" | cut -c1-200 | sed 's/^/     /'
    done
}

run_pipeline() {
    bash scripts/runpod_setup.sh || return 1
    export PATH="$HOME/.local/bin:$PATH"
    log "downloading to $RAW"
    uv run retina-dr download --raw-dir "$RAW" || return 1
    show_raw
    log "building to $OUT"
    uv run retina-dr build --raw-dir "$RAW" --out-dir "$OUT" || return 1
}

mkdir -p "$STATUS"
if [[ -d "$CODE/.git" ]]; then
    log "updating code"
    git -C "$CODE" pull --ff-only
else
    log "cloning code"
    git clone "$REPO_URL" "$CODE"
fi
cd "$CODE"

while true; do
    rm -f "$STATUS/DONE" "$STATUS/FAILED"
    log "running pipeline at commit $(git rev-parse --short HEAD)"
    if run_pipeline; then
        du -sh "$OUT" || true
        log "DONE"
        touch "$STATUS/DONE"
        exit 0
    fi
    git rev-parse --short HEAD > "$STATUS/FAILED"
    log "FAILED at commit $(git rev-parse --short HEAD). Waiting for a new commit to retry."
    current="$(git rev-parse HEAD)"
    until git fetch -q origin main && [[ "$(git rev-parse origin/main)" != "$current" ]]; do
        sleep "$POLL_SECONDS"
    done
    git pull -q --ff-only
    log "new commit $(git rev-parse --short HEAD) found, restarting the boot script"
    exec bash scripts/pod_bootstrap.sh
done
