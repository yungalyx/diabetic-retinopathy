# retina-dr: agent context

Read this file first in every session. It points to the rest of the project context.

## Purpose

Train a model that reads a fundus (retina) photo and outputs:

1. A binary label: referable diabetic retinopathy (ICDR grade 2 or higher), yes or no.
2. A severity grade 0-4 when the binary label is yes.

The model trains on EyePACS, APTOS 2019, and ODIR-5K. DDR is held out to measure performance on an unseen camera and population.

## Context files

| File | Content | Update rule |
|---|---|---|
| `docs/progress.md` | Current status, next actions, open questions | Update at the end of each work session |
| `docs/decisions.md` | Numbered decision log with the reason for each decision | Append a new entry. Do not edit old entries. Supersede them with a new entry |
| `docs/research/` | Research notes with sources | Add one file per research topic |
| `README.md` | How to run the pipeline | Update when commands or behavior change |

## Conventions

- Run `make check` before each commit. It runs ruff, mypy strict, and pytest.
- Write the test first. The test must fail before the change.
- Use ASD-STE100 Simplified Technical English for docs, commit messages, and procedures.
- Never write credentials to any file in this repository. Read the Kaggle token from `KAGGLE_API_TOKEN`.
- Never use the holdout source (DDR) to choose thresholds, epochs, or models.

## Layout

- `src/retina_dr/` holds the pipeline: `sources` (read labels) → `preprocess` (crop, resize, hash) → `pipeline` (dedupe, split, manifest).
- `scripts/runpod_setup.sh` prepares a RunPod pod.
- `data/` is not tracked. It holds `raw/` downloads and `processed/` images plus `manifest.parquet`.
