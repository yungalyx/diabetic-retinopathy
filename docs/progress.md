# Progress

Last update: 2026-10-21

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
| 7. Model comparison | Done. Frozen configuration: B3, 10 epochs, 3-seed ensemble (D-024, D-026, D-028). RETFound stopped |
| 8. Quality head, per-eye results, external test | Done 2026-10-16 except Messidor-2 (waits for ADCIS access). External result in D-036 |
| 9. RETFound comparison | Done 2026-10-21. RETFound at 224 px: test AUC 0.906 against 0.962. Frozen model stays primary (D-038 to D-040) |

## RunPod resources

| Resource | ID | Details | Cost |
|---|---|---|---|
| Pods | none | All pods are deleted (2026-10-21) | $0 |
| Network volume `retina-dr-data` | `5ac8gl5sy9` | 50 GB STANDARD, EU-RO-1, mounted at `/workspace` | About $3.50/month until deleted |
| Secret `kaggle_api_token` | `s7u36506fd` | Pod reads it as `KAGGLE_API_TOKEN` | Free |
| Secret `hf_token` | `jlzualo1qn` | Pod reads it as `HF_TOKEN` (RETFound weights) | Free |

The pod log is also written to `/workspace/bootstrap.log`.

## Next actions

Plan from the review council (D-027 to D-031).

Agent:
1. Read the quality run result (D-025).
2. Provenance in runs and builds; pin EyeQ (D-031).
3. Per-eye results and one referral rule (D-030).
4. Write the external analysis plan, rebuild at a tagged commit, retrain the frozen configuration, then run IDRiD and Messidor-2 once (D-029).

Alex and the researcher:
1. Clinic data: archive audit query, export format, de-identification path.
2. Start the macular edema (OCT) component plan.
3. Licence direction for research-only data and weights (D-015, D-020, D-027).
4. Clinician review of the DDR severe undergrades.

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

### Operating point and intervals (D-017, D-021, `metrics_ci.json`)

Threshold: probability 0.0317, set at 85% specificity on val. 95% intervals: 500 patient-level bootstrap resamples.

| Split | Sensitivity | Specificity | QWK tuned | VTDR sensitivity | Severe undergrading (grade 3-4 as 0-1) |
|---|---|---|---|---|---|
| test, all | 0.888 (0.865-0.906) | 0.853 (0.840-0.865) | 0.838 (0.818-0.854) | 0.967 | 15 of 215 (7.0%) |
| test, EyePACS | 0.880 (0.856-0.904) | 0.844 (0.830-0.860) | 0.815 (0.794-0.835) | 0.963 | 14 of 160 (8.8%) |
| test, APTOS | 0.972 (0.943-1.000) | 0.902 (0.860-0.938) | 0.917 (0.892-0.940) | 0.982 | 1 of 55 (1.8%) |
| test, ODIR | 0.845 (0.780-0.899) | 0.880 (0.853-0.907) | n/a | n/a | n/a |
| DDR holdout | 0.788 (0.778-0.798) | 0.948 (0.943-0.954) | 0.764 (0.753-0.775) | 0.801 | 193 of 1,146 (16.8%) |

The val threshold gives 95% specificity on DDR, not 85%. The model gives lower scores on DDR images. Sensitivity falls to 0.79.

DDR confusion matrix (rows: true grade, columns: predicted grade 0-4):

| True | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 0 | 6,036 | 91 | 9 | 2 | 3 |
| 1 | 290 | 241 | 83 | 2 | 11 |
| 2 | 835 | 777 | 1,633 | 1,024 | 205 |
| 3 | 4 | 6 | 22 | 164 | 40 |
| 4 | 121 | 62 | 77 | 153 | 497 |

Findings:
1. Grade 2 spreads over predicted grades 0-4. 36% of DDR grade 2 is predicted as 0 or 1. This agrees with a different grade 2 boundary between DDR and EyePACS graders.
2. 121 of 910 DDR grade 4 images are predicted as grade 0. Grade 3 has only 4. This is a specific failure. Possible causes: laser scars after treatment, vitreous hemorrhage, or label errors. Inspect these images.

### DDR grade 4 predicted as grade 0 (2026-10-15, `scripts/analysis/ddr_grade4_failures.py`)

The model is confident on these 121 images: median referral probability 0.0003, maximum 0.067. They are not borderline cases.
A visual check of 64 random thumbnails (128 px, by the agent, not a clinician) gave these groups:

| Group | Count of 64 | Example look |
|---|---|---|
| Retina hidden by haze (probably vitreous hemorrhage or media opacity) | about 20 | Blurred, green or yellow image, few visible vessels |
| Large hemorrhage | about 12 | Dark red areas in front of the retina |
| Retina looks normal or mild at 128 px | about 19 | Needs full-resolution review. Possible small new vessels or label errors |
| Laser scars or dense exudates | about 5 | Many round pale spots |
| Fibrous tissue or traction | about 5 | Pale membranes along vessels |
| Crop failure: white corners | 2 | `crop_fundus` did not remove a white background. Fixed in D-022. Only 25 of 57,952 images are affected |

Image statistics (mean over up to 150 images per group): the failures are darker (5.8% dark pixels, against 1.3% for correct grade 0) and less sharp than grade 0.
The correctly graded grade 4 images are also blurred, so image quality alone does not explain the failure.
The grade 4 image names in DDR use the pattern `007-NNNN`. The grade 0 names use a date. DDR grade 4 can come from a different collection.

Conclusions:
1. The model has no "cannot grade" output. It calls an image with a hidden retina grade 0 with high confidence. This is the gradability risk in the study protocol.
2. Advanced disease (large hemorrhage, scars, fibrous tissue) is rare in training. EyePACS grade 4 is about 2%.
3. About a third of the failures need full-resolution review by a clinician before we call them model errors.

## Open questions

- For the researcher: research-only data and weights (DDR, RETFound, Kaggle competition data) are used now. Decide how to handle them for a commercial model (D-020).
- ODIR-5K licence: find the original release and its terms.

- Are the normal-looking DDR grade 4 images label errors? A clinician must review them at full resolution.

## Later improvements

- Decode JPEGs at reduced size (Pillow draft mode) to cut a full rebuild from about 95 minutes to about 20.
- Combine both eyes of one patient (EyePACS, ODIR). This gave about +0.02 kappa in 2015.
- Soft pseudo-labels on unlabeled images.
