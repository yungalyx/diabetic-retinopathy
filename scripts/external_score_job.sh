#!/usr/bin/env bash
# GPU pod job: score the frozen model on the external sets once (D-029, D-035).
# Run only after docs/analysis-plan-external.sha256 is committed. Stops the pod at the end.
set -uo pipefail
EXP=/workspace/runs/exp
OUT=/workspace/runs/external/b3-e10-ens3-q
log() { echo "[score $(date -u +%H:%M:%S)] $*"; }
export UV_PROJECT_ENVIRONMENT=/root/venv UV_CACHE_DIR=/workspace/.uv-cache UV_LINK_MODE=copy
export PATH="$HOME/.local/bin:$PATH"

cd /code
log "commit $(git rev-parse HEAD)"
if [[ -n "${PLAN_B64:-}" ]]; then
    mkdir -p /workspace/private
    echo "$PLAN_B64" | base64 -d | gunzip > /workspace/private/analysis-plan-external.md
    expected="$(cut -d' ' -f1 docs/analysis-plan-external.sha256 | head -1)"
    actual="$(sha256sum /workspace/private/analysis-plan-external.md | cut -d' ' -f1)"
    log "plan stored on the volume. expected $expected actual $actual"
    if [[ "$expected" != "$actual" ]]; then log "PLAN HASH MISMATCH: not scoring"; exit 1; fi
fi
log "plan hash file: $(cat docs/analysis-plan-external.sha256 | head -1)"
bash scripts/runpod_setup.sh >/dev/null 2>&1 || log "setup returned an error"
if [[ -f "$OUT/metrics.json" ]]; then
    log "already scored: $OUT/metrics.json exists. Not scoring again (D-029)."
else
    uv run retina-dr score-external --data-dir /workspace/external \
        --run-dirs "$EXP/b3-e10" "$EXP/b3-e10-s1" "$EXP/b3-e10-s2" \
        --quality-run "$EXP/quality-b3" --ensemble-run "$EXP/b3-e10-ens3-q" \
        --out-dir "$OUT" </dev/null
fi
log "DONE"
if [[ -n "${RUNPOD_POD_ID:-}" ]] && command -v runpodctl >/dev/null 2>&1; then
    runpodctl stop pod "$RUNPOD_POD_ID" || true
fi
