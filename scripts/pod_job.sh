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
export HF_HUB_CACHE="${HF_HUB_CACHE:-/workspace/.hf-cache}"

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

log "results so far, and HuggingFace access"
uv run python scripts/analysis/probe_runs_and_hf.py </dev/null || log "probe failed"

EYEQ="/workspace/labels/eyeq_train.csv"
EYEQ_URL="https://raw.githubusercontent.com/HzFu/EyeQ/1dcfabb08cd31045de1a4a3ee82e48f6833697aa/data/Label_EyeQ_train.csv"
EYEQ_SHA256="34f39f8923f62a13fcaeccef7e5da7ec607b45805e7736042fa274b2731b9480"
mkdir -p "$(dirname "$EYEQ")"
if [[ ! -f "$EYEQ" ]]; then
    curl -fsSL "$EYEQ_URL" -o "$EYEQ.part" && mv "$EYEQ.part" "$EYEQ" || log "EyeQ download failed"
fi
if [[ -f "$EYEQ" ]] && echo "$EYEQ_SHA256  $EYEQ" | sha256sum -c --quiet; then
    log "adding pinned EyeQ quality labels to the local manifest (D-025, D-031)"
    uv run retina-dr add-quality --data-dir "$LOCAL" --eyeq-csv "$EYEQ" </dev/null \
        || log "add-quality failed: runs train without quality labels"
    cp "$LOCAL/manifest.parquet" "/workspace/labels/manifest_with_quality.parquet"
else
    log "EyeQ labels missing or checksum mismatch: runs train without quality labels"
fi

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
    git rev-parse HEAD > "$run/COMMIT"
    log "finished $name"
    uv run python -c "import json,sys; [print('RESULT', sys.argv[1], r) for r in json.load(open(sys.argv[2]))['rows'] if r['source']=='all']" "$name" "$run/metrics.json" </dev/null || true
done

grep -v '^\s*#' scripts/ensembles.txt | grep -v '^\s*$' | while read -r name members; do
    out="$RUNS/exp/$name"
    [[ -f "$out/metrics.json" ]] && continue
    dirs=()
    extra=()
    for member in $members; do
        if [[ "$member" == quality=* ]]; then
            extra=(--quality-run "$RUNS/exp/${member#quality=}")
            [[ -f "$RUNS/exp/${member#quality=}/predictions.parquet" ]] || continue 2
        else
            dirs+=("$RUNS/exp/$member")
        fi
    done
    for dir in "${dirs[@]}"; do [[ -f "$dir/predictions.parquet" ]] || continue 2; done
    log "ensemble $name from $members"
    uv run retina-dr ensemble --run-dirs "${dirs[@]}" "${extra[@]}" --out-dir "$out" </dev/null \
        || log "FAILED ensemble $name"
    git rev-parse HEAD > "$out/COMMIT"
done

if [[ -n "${RUNPOD_POD_ID:-}" ]] && command -v runpodctl >/dev/null 2>&1; then
    log "stopping pod $RUNPOD_POD_ID"
    runpodctl stop pod "$RUNPOD_POD_ID" || log "runpodctl stop failed; stop the pod manually"
fi
