#!/usr/bin/env bash
# GPU pod job: score the frozen model once on the DeepDRiD ultra-widefield images (D-042).
# Builds the set, scores it, runs the tuning rehearsal, prints the results, and stops the pod.
set -uo pipefail
RAW=/workspace/external_raw_uwf
DATA=/workspace/external_uwf
EXP=/workspace/runs/exp
OUT=/workspace/runs/external_uwf/b3-e10-ens3-q
log() { echo "[uwf $(date -u +%H:%M:%S)] $*"; }
export UV_PROJECT_ENVIRONMENT=/root/venv UV_CACHE_DIR=/workspace/.uv-cache UV_LINK_MODE=copy
export PATH="$HOME/.local/bin:$PATH"

cd /code
log "commit $(git rev-parse HEAD)"
bash scripts/runpod_setup.sh >/dev/null 2>&1 || log "setup returned an error"
if [[ ! -d "$RAW/deepdrid_uwf" ]]; then
    log "DeepDRiD ultra-widefield images from the official GitHub repository"
    mkdir -p "$RAW"
    git clone -q --depth 1 --filter=blob:none --sparse https://github.com/deepdrdoc/DeepDRiD.git "$RAW/deepdrid_uwf"
    git -C "$RAW/deepdrid_uwf" sparse-checkout set "ultra-widefield_images"
    git -C "$RAW/deepdrid_uwf" rev-parse HEAD > "$RAW/deepdrid_uwf.COMMIT"
fi
if [[ ! -f "$DATA/manifest.parquet" ]]; then
    uv run retina-dr build-external --raw-dir "$RAW" --out-dir "$DATA" </dev/null
fi
uv run python -c "
import polars as pl
m = pl.read_parquet('$DATA/manifest.parquet')
print(m.group_by('grade').len().sort('grade'))
print('eyes', m['eye_id'].n_unique(), 'patients', m['patient_id'].n_unique())
" </dev/null
sha256sum "$DATA/manifest.parquet"
if [[ -f "$OUT/metrics.json" ]]; then
    log "already scored: $OUT/metrics.json exists. Not scoring again (D-042)."
else
    uv run retina-dr score-external --data-dir "$DATA" \
        --run-dirs "$EXP/b3-e10" "$EXP/b3-e10-s1" "$EXP/b3-e10-s2" \
        --quality-run "$EXP/quality-b3" --ensemble-run "$EXP/b3-e10-ens3-q" \
        --out-dir "$OUT" </dev/null >/dev/null
fi
uv run python -c "
import json
m = json.load(open('$OUT/metrics.json'))
for r in m['rows']:
    print('ROW', json.dumps(r))
print('UNGRADABLE', m.get('ungradable'))
" </dev/null
uv run retina-dr rehearse-tuning --predictions "$OUT/predictions.parquet" \
    --frozen-metrics "$EXP/b3-e10-ens3-q/metrics.json" \
    --out-dir /workspace/runs/rehearsal_uwf/b3-e10-ens3-q --splits 1000 --seed 0 </dev/null
log "DONE"
if [[ -n "${RUNPOD_POD_ID:-}" ]] && command -v runpodctl >/dev/null 2>&1; then
    runpodctl stop pod "$RUNPOD_POD_ID" || true
fi
