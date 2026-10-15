# Progress

Last update: 2026-10-15

## Status

| Step | Status |
|---|---|
| 1. Choose datasets, model design, split plan | Done (D-001 to D-010) |
| 2. Download and build pipeline | Done. 69 tests |
| 3. Research Kaggle post-mortems | Done (docs/research/kaggle-postmortems.md) |
| 4. RunPod access for the agent | Done. Code is on GitHub (yungalyx/diabetic-retinopathy) |
| 5. Run download and build on a pod | Done 2026-10-15 07:15 UTC at commit 2ca78e1. 57,633 images in the manifest |
| 6. Training script (EfficientNet-B3, two heads) | Done (D-013, D-014). 97 tests |
| 6b. Baseline training run | Done 2026-10-15 16:31 UTC, run `b3-20261015-1520-603e880`, best epoch 9 |
| 7. RETFound experiment | Not started. Research-only weights (D-015, D-020) |

## RunPod resources

| Resource | ID | Details | Cost |
|---|---|---|---|
| Pod `retina-dr-4090` | `00difbrelhsfon` | Build pod. RTX 4090, EU-RO-1, 250 GB container disk. STOPPED. Its host GPU was taken, so it could not restart | $5.52 billed |
| Pod `retina-dr-train-4090` | `rqcktnogjy5zqf` | Training pod. RTX 4090, EU-RO-1, 60 GB container disk. STOPPED by the job after the run | $0.89/hr while running. $1.22 billed at last read (billing lags) |
| Network volume `retina-dr-data` | `5ac8gl5sy9` | 50 GB STANDARD, EU-RO-1, mounted at `/workspace` | About $3.50/month until deleted |
| Secret `kaggle_api_token` | `s7u36506fd` | Pod reads it as `KAGGLE_API_TOKEN` | Free |

The pod log is also written to `/workspace/bootstrap.log`.

## Next actions

Work continues with research-only data and weights (D-020).

1. Alex: decide if this repo stays public. The study protocol is in `docs/private/` and is not committed.
2. Agent: run `retina-dr evaluate` on the baseline run (D-017, D-021). The code is done.
3. Agent: add IDRiD and Messidor-2 as the external test (D-016), and provenance records (D-019).
4. Agent: RETFound experiment (step 7) with a 10-epoch schedule.

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

## Baseline result (run b3-20261015-1520-603e880, commit 603e880)

EfficientNet-B3, 512 px, 15 epochs, best epoch 9 by val score. 8x TTA. Cutpoints tuned on val: 0.43 / 1.31 / 2.22 / 3.29.
Files: `/workspace/runs/b3-20261015-1520-603e880/` (`best.pt`, `metrics.json`, `predictions.parquet`, `history.jsonl`).

| Split | Source | Images | AUC (grade 2+) | Sens at 90% spec | Sens at 95% spec | QWK fixed | QWK tuned |
|---|---|---|---|---|---|---|---|
| val | all | 4,591 | 0.963 | 0.914 | 0.844 | 0.850 | 0.855 |
| test | all | 4,568 | 0.939 | 0.871 | 0.823 | 0.833 | 0.838 |
| test | EyePACS | 3,532 | 0.932 | 0.862 | 0.807 | 0.807 | 0.815 |
| test | APTOS | 345 | 0.982 | 0.972 | 0.936 | 0.920 | 0.917 |
| test | ODIR | 691 | 0.935 | 0.831 | 0.770 | n/a | n/a |
| holdout | DDR | 12,388 | 0.914 | 0.840 | 0.785 | 0.749 | 0.764 |

Generalisation gap (test to DDR holdout): AUC -0.025, QWK tuned -0.074.
Val curve: AUC 0.928 (epoch 0), 0.950 (2), 0.954 (5), 0.958 (8), 0.958 (9), 0.954 (14). Train loss fell from 0.73 to 0.10, so epochs 10-14 overfit.
Speed: about 185 images/s, 3.5 minutes per epoch, GPU at 97%.

## Open questions

- For the researcher: research-only data and weights (DDR, RETFound, Kaggle competition data) are used now. Decide how to handle them for a commercial model (D-020).
- ODIR-5K licence: find the original release and its terms.

- Does the severity gap on DDR come from the model or from grader differences between sources? Check the DDR confusion matrix in predictions.parquet.

## Later improvements

- Decode JPEGs at reduced size (Pillow draft mode) to cut a full rebuild from about 95 minutes to about 20.
- Combine both eyes of one patient (EyePACS, ODIR). This gave about +0.02 kappa in 2015.
- Soft pseudo-labels on unlabeled images.
