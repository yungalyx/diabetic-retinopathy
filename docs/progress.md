# Progress

Last update: 2026-10-14

## Status

| Step | Status |
|---|---|
| 1. Choose datasets, model design, split plan | Done (D-001 to D-010) |
| 2. Download and build pipeline | Done. 69 tests |
| 3. Research Kaggle post-mortems | Done (docs/research/kaggle-postmortems.md) |
| 4. RunPod access for the agent | Done. Code is on GitHub (yungalyx/diabetic-retinopathy) |
| 5. Run download and build on a pod | Done 2026-10-15 07:15 UTC at commit 2ca78e1. 57,633 images in the manifest |
| 6. Training script (EfficientNet-B3, two heads) | Done (D-013). 92 tests, CPU end-to-end test passes |
| 7. RETFound experiment | Not started |

## RunPod resources

| Resource | ID | Details | Cost |
|---|---|---|---|
| Pod `retina-dr-4090` | `00difbrelhsfon` | RTX 4090, Secure Cloud, EU-RO-1, template `runpod-torch-v280`, 250 GB container disk. STOPPED (EXITED) | $0.89/hr while running. $4.64 billed for the build |
| Network volume `retina-dr-data` | `5ac8gl5sy9` | 50 GB STANDARD, EU-RO-1, mounted at `/workspace` | About $3.50/month until deleted |
| Secret `kaggle_api_token` | `s7u36506fd` | Pod reads it as `KAGGLE_API_TOKEN` | Free |

The pod log is also written to `/workspace/bootstrap.log`.

## Next actions

1. Agent: start the pod. The boot script skips the build and runs `scripts/pod_job.sh`: a smoke run, then the full run.
2. Agent: read `/workspace/runs/<run>/metrics.json` from the log. Report val, test, and DDR holdout results.

## Verified source facts (pod log, 2026-10-15)

| Source | Label file | Columns used | Labelled images |
|---|---|---|---|
| EyePACS | `trainLabels.csv` | `image`, `level` | 35,126 (train only; the 53,576 test images have no labels) |
| APTOS | `train.csv` | `id_code`, `diagnosis` | 3,662 |
| DDR | `DR_grading.csv` (Kaggle mirror, flat image folder) | `id_code`, `diagnosis` | 12,522 |
| ODIR | `full_df.csv` | `ID`, `Left-/Right-Fundus`, `Left-/Right-Diagnostic Keywords` | 6,642 (74 ungradable eyes dropped) |

## Build result (2026-10-15, commit 2ca78e1)

`/workspace/processed/manifest.parquet`: 57,633 images, 2.9 GB of 512 px JPEGs.

| Source | Split | Images | Positive rate (grade 2+) |
|---|---|---|---|
| APTOS | train / val / test | 2,802 / 339 / 345 | 0.385 / 0.389 / 0.409 |
| EyePACS | train / val / test | 28,011 / 3,582 / 3,532 | 0.194 / 0.198 / 0.204 |
| ODIR | train / val / test | 5,273 / 670 / 691 | 0.171 / 0.170 / 0.214 |
| DDR | holdout | 12,388 | 0.454 |

Duplicates: 264 groups covering 544 images. 39 groups had conflicting labels and were dropped. 319 images dropped in total.
Nearest-neighbour RMS distance: 0-1: 516, 1-2: 28, 2-4: 684, 4-8: 23,643, 8-16: 29,774, 16-32: 3,232, 32+: 75.

## Open questions

- None for the data stage.

## Later improvements

- Decode JPEGs at reduced size (Pillow draft mode) to cut a full rebuild from about 95 minutes to about 20.
- Combine both eyes of one patient (EyePACS, ODIR). This gave about +0.02 kappa in 2015.
- Soft pseudo-labels on unlabeled images.
