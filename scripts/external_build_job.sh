#!/usr/bin/env bash
# CPU pod job: download the external sets, build them, and freeze the model record (D-035).
# Needs KAGGLE_API_TOKEN and the network volume at /workspace. Does not score anything.
set -uo pipefail
RAW=/workspace/external_raw
OUT=/workspace/external
log() { echo "[external $(date -u +%H:%M:%S)] $*"; }

pip install -q polars pillow numpy fastexcel kaggle
cd /code
mkdir -p "$RAW/idrid" "$RAW/messidor2_grades"
if [[ -z "$(ls -A "$RAW/idrid")" ]]; then
    log "IDRiD from Kaggle (CC BY 4.0 mirror)"
    kaggle datasets download aaryapatel98/indian-diabetic-retinopathy-image-dataset -p "$RAW/idrid" --unzip
fi
if [[ ! -d "$RAW/deepdrid" ]]; then
    log "DeepDRiD evaluation set from the official GitHub repository"
    git clone -q --depth 1 --filter=blob:none --sparse https://github.com/deepdrdoc/DeepDRiD.git "$RAW/deepdrid"
    git -C "$RAW/deepdrid" sparse-checkout set "regular_fundus_images/Online-Challenge1&2-Evaluation"
    git -C "$RAW/deepdrid" rev-parse HEAD > "$RAW/deepdrid.COMMIT"
fi
if [[ -z "$(ls -A "$RAW/messidor2_grades")" ]]; then
    log "Messidor-2 grades (Google Brain). Images wait for ADCIS access."
    kaggle datasets download google-brain/messidor2-dr-grades -p "$RAW/messidor2_grades" --unzip
fi
log "IDRiD files:"; find "$RAW/idrid" -name "*.csv" | sed 's/^/  /'
log "Messidor-2 grade files:"; find "$RAW/messidor2_grades" -type f | sed 's/^/  /'
head -3 "$(find "$RAW/messidor2_grades" -name "*.csv" | head -1)"

log "building external set at $OUT"
PYTHONPATH=src python -u -m retina_dr.cli build-external --raw-dir "$RAW" --out-dir "$OUT"
PYTHONPATH=src python -u -c "
import polars as pl
m = pl.read_parquet('$OUT/manifest.parquet')
print(m.group_by('source', 'grade').len().sort('source', 'grade'))
print(m.group_by('source', 'quality').len().sort('source', 'quality'))
print('eyes', m.group_by('source').agg(pl.col('eye_id').n_unique()).rows())
"
sha256sum "$OUT/manifest.parquet"
python -u scripts/freeze_model.py b3-e10-ens3-q
log "DONE"
