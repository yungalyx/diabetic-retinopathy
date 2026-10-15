# retina-dr

This project merges four public fundus image datasets into one preprocessed image set.
It also writes one manifest file for diabetic retinopathy (DR) model training.
The manifest supports two model heads: a binary "referable DR" head and a severity head.

## Sources

| Source | Kaggle | Labels used |
|---|---|---|
| EyePACS | competition `diabetic-retinopathy-detection` | ICDR grade 0-4 |
| APTOS 2019 | competition `aptos2019-blindness-detection` | ICDR grade 0-4 |
| DDR | dataset `mariaherrerot/ddrdataset` | ICDR grade 0-4 (grade 5 ungradable is dropped) |
| ODIR-5K | dataset `andrewmvd/ocular-disease-recognition-odir5k` | Binary label only, from diagnostic keywords |

## Manifest columns

The build writes `data/processed/manifest.parquet`. Each row is one image.

| Column | Meaning |
|---|---|
| `path` | Image path relative to the output folder |
| `source` | `eyepacs`, `aptos`, `ddr`, or `odir` |
| `image_id` | Image name in the source |
| `patient_id` | Patient key. APTOS and DDR give no patient ID, so each image is its own patient |
| `grade` | ICDR grade 0-4 for the severity head. Null for ODIR |
| `binary_label` | 1 when the grade is at or above the cutoff (default 2, referable DR) |
| `split` | `train`, `val`, `test`, or `holdout` |
| `dup_group` | Duplicate group id. Images in one group are copies of one photo |

Split rules:

- All images of the holdout source (default DDR) go to `holdout`.
- Other images go to `train` (80%), `val` (10%), or `test` (10%) by a hash of the patient ID.
- All images of one patient get the same split.
- Two images are duplicates when their 32x32 grayscale fingerprints differ by RMS 2 or less (0-255 scale).
- When duplicates have the same labels, the build keeps one copy. The holdout copy has priority.
- When duplicates have different labels, the build drops all copies.
- `all_records.parquet` holds every row before duplicate removal.

## Run on RunPod

Prerequisites:

1. Accept the rules of both Kaggle competitions in a browser:
   - https://www.kaggle.com/c/diabetic-retinopathy-detection/rules
   - https://www.kaggle.com/c/aptos2019-blindness-detection/rules
2. Create a Kaggle API token at https://www.kaggle.com/settings (section "API").
3. Create a pod with a volume of 250 GB or more.
4. Store the Kaggle token as a RunPod secret. Add `KAGGLE_API_TOKEN` to the pod environment and set its value to that secret.

Automatic procedure: start the pod with this command. The script clones the repo, downloads, and builds. It writes `/workspace/status/DONE` or `/workspace/status/FAILED`.

```bash
curl -fsSL https://raw.githubusercontent.com/yungalyx/diabetic-retinopathy/main/scripts/pod_bootstrap.sh | bash
```

Manual procedure:

1. Copy this folder to the pod.
2. Run `bash scripts/runpod_setup.sh`.
3. Run `uv run retina-dr download --raw-dir data/raw`.
4. Run `uv run retina-dr build --raw-dir data/raw --out-dir data/processed`.
5. Read the summary table. Make sure each source has the expected image count.
6. Copy `data/processed` to network storage.
7. Delete `data/raw` to stop paying for the raw storage.

The download removes archives after extraction. Use `--keep-archives` to keep them.
The download needs about 1.5 times more disk when you keep archives.

Each command accepts `--sources`, for example `--sources aptos,odir`.
The build accepts `--binary-cutoff`, `--holdout`, `--seed`, `--size`, and `--workers`.
A second build reuses images that already exist in the output folder.

## Verify on first run

These items come from public descriptions and are not yet verified against the real files:

- The DDR Kaggle slug and its folder layout (`DR_grading/*.txt`).
- The ODIR file `full_df.csv` and its column names.
- The EyePACS test label file `retinopathy_solution.csv`. When it is missing, the build uses training images only.

When a label file has different columns, the build stops with an error that names the missing columns.

## Development

Run `make check`. This command runs ruff, mypy (strict), and pytest.
Run `make fmt` to fix formatting.

## Train

Run `uv run retina-dr train --data-dir data/processed --run-dir runs/<name>`.
The run folder gets `config.json`, `history.jsonl` (one line per epoch), `best.pt`, `predictions.parquet`, and `metrics.json`.
Use `--limit 600 --epochs 1` for a smoke run. See D-013 in `docs/decisions.md` for the default settings.

On the pod, `scripts/pod_job.sh` runs after the build. It copies the processed set to local disk and runs a smoke run and the full run.
It stops the pod when the full run finishes.
