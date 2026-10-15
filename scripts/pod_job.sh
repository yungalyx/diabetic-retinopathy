#!/usr/bin/env bash
# Job that runs after the data build: copy the processed set to local disk, run a short
# smoke training, then the full training. Stop the pod when the full run finishes.
set -euo pipefail

SRC="/workspace/processed"
LOCAL="/root/processed"
RUNS="/workspace/runs"
COMMIT="$(git rev-parse --short HEAD)"
STAMP="$(date -u +%Y%m%d-%H%M)"

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/root/venv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.uv-cache}"
export UV_LINK_MODE=copy

log() { echo "[job $(date -u +%H:%M:%S)] $*"; }

if [[ ! -f "$LOCAL/manifest.parquet" ]]; then
    log "copying $SRC to local disk $LOCAL"
    mkdir -p "$LOCAL"
    cp -a "$SRC/." "$LOCAL/"
fi
log "local data: $(du -sh "$LOCAL" | cut -f1)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

log "smoke run (1 epoch, 600 rows per split)"
uv run retina-dr train --data-dir "$LOCAL" --run-dir "$RUNS/smoke-$STAMP-$COMMIT" \
    --epochs 1 --limit 600 --log-every 10

log "full run"
uv run retina-dr train --data-dir "$LOCAL" --run-dir "$RUNS/b3-$STAMP-$COMMIT" \
    --epochs 15 --log-every 300
log "full run finished: $RUNS/b3-$STAMP-$COMMIT"

if [[ -n "${RUNPOD_POD_ID:-}" ]] && command -v runpodctl >/dev/null 2>&1; then
    log "stopping pod $RUNPOD_POD_ID"
    runpodctl stop pod "$RUNPOD_POD_ID" || log "runpodctl stop failed; stop the pod manually"
fi
