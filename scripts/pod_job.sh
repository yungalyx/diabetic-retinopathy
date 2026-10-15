#!/usr/bin/env bash
# Job that runs after the data build: copy the processed set to local disk, then run each
# experiment in scripts/experiments.txt (a smoke run, then the full run). Stop the pod at the end.
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

ARCHIVE="/workspace/processed.tar"
if [[ ! -f "$LOCAL/manifest.parquet" ]]; then
    if [[ -f "$ARCHIVE" && "$ARCHIVE" -nt "$SRC/manifest.parquet" ]]; then
        log "extracting $ARCHIVE to $LOCAL"
        tar -xf "$ARCHIVE" -C "$(dirname "$LOCAL")"
    else
        log "copying $SRC to local disk $LOCAL (slow: many small files)"
        mkdir -p "$LOCAL"
        cp -a "$SRC/." "$LOCAL/"
        log "writing $ARCHIVE so later runs copy one file"
        tar -cf "$ARCHIVE" -C "$(dirname "$LOCAL")" "$(basename "$LOCAL")"
    fi
fi
log "local data: $(du -sh "$LOCAL" | cut -f1)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

EXPERIMENTS="scripts/experiments.txt"
grep -v '^\s*#' "$EXPERIMENTS" | grep -v '^\s*$' | while read -r name args; do
    run="$RUNS/exp/$name"
    if [[ -f "$run/metrics.json" ]]; then
        log "skip $name: $run/metrics.json exists"
        continue
    fi
    log "smoke run for $name (1 epoch, 600 rows per split)"
    # shellcheck disable=SC2086
    uv run retina-dr train --data-dir "$LOCAL" --run-dir "$RUNS/smoke/$name" \
        $args --epochs 1 --limit 600 --log-every 10 </dev/null || { log "FAILED smoke $name"; continue; }
    log "full run $name at commit $COMMIT"
    # shellcheck disable=SC2086
    uv run retina-dr train --data-dir "$LOCAL" --run-dir "$run" $args --log-every 300 </dev/null \
        || { log "FAILED $name"; continue; }
    echo "$COMMIT" > "$run/COMMIT"
    log "finished $name"
    uv run python -c "import json,sys; [print('RESULT', sys.argv[1], r) for r in json.load(open(sys.argv[2]))['rows'] if r['source']=='all']" "$name" "$run/metrics.json" </dev/null || true
done

if [[ -n "${RUNPOD_POD_ID:-}" ]] && command -v runpodctl >/dev/null 2>&1; then
    log "stopping pod $RUNPOD_POD_ID"
    runpodctl stop pod "$RUNPOD_POD_ID" || log "runpodctl stop failed; stop the pod manually"
fi
