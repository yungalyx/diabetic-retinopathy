# retina-dr: agent context

Read this file first in every session. It points to the rest of the project context.

## Purpose

Train a model that reads a fundus (retina) photo and outputs:

1. A binary label: referable diabetic retinopathy (ICDR grade 2 or higher), yes or no.
2. A severity grade 0-4 when the binary label is yes.

The model trains on EyePACS, APTOS 2019, and ODIR-5K. DDR is a development holdout (unseen camera and population). IDRiD and Messidor-2 are the untouched external test (D-016).

All models in this repo are research-only. See D-015 and `docs/research/dataset-licences.md`.

This repo is the public-data DR component of a larger research study. Study context that is not public is in `docs/private/` (not tracked). Read it when it exists.

## Context files

| File | Content | Update rule |
|---|---|---|
| `docs/progress.md` | Current status, next actions, open questions | Update at the end of each work session |
| `docs/worklog.md` | Dated log of what was done, the result, and the cost | Append an entry at the end of each work block |
| `docs/reports/` | Reports for the researcher, with figures | Add one file per report |
| `docs/decisions.md` | Numbered decision log with the reason for each decision | Append a new entry. Do not edit old entries. Supersede them with a new entry |
| `docs/research/` | Research notes with sources | Add one file per research topic |
| `README.md` | How to run the pipeline | Update when commands or behavior change |

## Conventions

- Run `make check` before each commit. It runs ruff, mypy strict, and pytest.
- Write the test first. The test must fail before the change.
- Use ASD-STE100 Simplified Technical English for docs, commit messages, and procedures.
- Never write credentials to any file in this repository. Read the Kaggle token from `KAGGLE_API_TOKEN`.
- Never use DDR to choose thresholds or epochs. Use DDR only to compare finished models (D-016).
- Never use IDRiD or Messidor-2 for any choice. Evaluate them once, on the frozen model.
- Never commit files from `docs/private/`. They hold an unpublished study protocol.

## Layout

- `src/retina_dr/` holds the pipeline: `sources` (read labels) → `preprocess` (crop, resize, hash) → `pipeline` (dedupe, split, manifest).
- `scripts/runpod_setup.sh` prepares a RunPod pod.
- `data/` is not tracked. It holds `raw/` downloads and `processed/` images plus `manifest.parquet`.
