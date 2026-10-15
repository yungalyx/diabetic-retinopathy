#!/usr/bin/env bash
# Run at pod boot: get the code, download the data, and build the processed set.
# Raw data goes to the container disk (temporary). Processed data goes to the network volume.
# After each run, wait for a new commit that changes code, pull it, and run again.
# Raw downloads stay on the container disk between runs. Stop the pod to end billing.
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
    if [[ ! -f "$OUT/manifest.parquet" || -n "${REBUILD:-}" ]]; then
        log "downloading to $RAW"
        uv run retina-dr download --raw-dir "$RAW" || return 1
        show_raw
        log "building to $OUT"
        uv run retina-dr build --raw-dir "$RAW" --out-dir "$OUT" || return 1
    else
        log "manifest exists at $OUT, skipping download and build (set REBUILD=1 to force)"
    fi
    if [[ -f scripts/pod_job.sh ]]; then
        log "running scripts/pod_job.sh"
        bash scripts/pod_job.sh || return 1
    fi
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

wait_for_code_commit() {
    local current changed
    current="$(git rev-parse HEAD)"
    while true; do
        sleep "$POLL_SECONDS"
        git fetch -q origin main || continue
        [[ "$(git rev-parse origin/main)" == "$current" ]] && continue
        changed="$(git diff --name-only "$current" origin/main -- src scripts pyproject.toml uv.lock)"
        git pull -q --ff-only
        current="$(git rev-parse HEAD)"
        [[ -n "$changed" ]] && return 0
        log "new commit $(git rev-parse --short HEAD) changes no code, still waiting"
    done
}

rm -f "$STATUS/DONE" "$STATUS/FAILED"
log "running pipeline at commit $(git rev-parse --short HEAD)"
if run_pipeline; then
    du -sh "$OUT" || true
    git rev-parse --short HEAD > "$STATUS/DONE"
    log "DONE at commit $(git rev-parse --short HEAD). Waiting for a code commit to run again."
else
    git rev-parse --short HEAD > "$STATUS/FAILED"
    log "FAILED at commit $(git rev-parse --short HEAD). Waiting for a code commit to retry."
fi
wait_for_code_commit
log "code commit $(git rev-parse --short HEAD) found, restarting the boot script"
exec bash scripts/pod_bootstrap.sh
