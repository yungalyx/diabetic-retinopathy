# Work log

One entry per work block, newest last. Each entry states what was done, the result, and the cost.
Decisions are in `docs/decisions.md`. Current status is in `docs/progress.md`.

## 2026-10-14: Plan and data pipeline

- Chose the sources (EyePACS, APTOS, DDR, ODIR), the two-head model, and the split plan (D-001 to D-010).
- Collected findings from past Kaggle solutions in `docs/research/kaggle-postmortems.md`.
- Wrote the download and build pipeline: label readers, fundus crop, resize to 512 px, duplicate removal, patient-level split.
- Set up RunPod access for the agent and a public GitHub repo for the pod to clone.

## 2026-10-15 (morning): Build on RunPod

- Ran download and build on an RTX 4090 pod. Result: 57,633 images in the manifest.
- Fixed: EyePACS `sample/` copies crashed the image index. The reader now excludes known copy folders.
- Fixed: the DDR Kaggle mirror has a CSV, not `.txt` label lists. The reader now accepts both.
- Fixed: the 64-bit dhash matched most fundus photos and dropped 25,303 images. Duplicates now use a 32x32 fingerprint with RMS distance 2 or less (D-011, D-012).
- Cost: $5.52 for the build pod.

## 2026-10-15 (midday): Baseline training

- Wrote the training code: EfficientNet-B3, binary and severity heads, decayed rebalancing, 8x test-time augmentation (D-013).
- Fixed: training ran at 35 images/s with the GPU idle. Augmentation moved to PIL without hue, and the loader reads the cgroup CPU count. Speed rose to about 185 images/s (D-014).
- Fixed: copying 57,000 small files from the network volume took 17 minutes. The processed set is now one tar file.
- Result: test AUC 0.939, DDR AUC 0.914, DDR kappa 0.764. Best epoch 9 of 15.
- Cost: about $1.90 for the training pod.

## 2026-10-15 (afternoon): Researcher context and new metrics

- Read the researcher's dataset list and protocol primer. Recorded licences in `docs/research/dataset-licences.md`.
- Recorded D-015 to D-021: research-only status, DDR as a development holdout, IDRiD and Messidor-2 as the external test, new metrics, ungradable images, provenance, and operating point.
- Kept the protocol summary in `docs/private/` (not committed), because the repo is public.
- Added `retina-dr evaluate`: operating point at 85% val specificity, vision-threatening DR sensitivity, severe undergrading, and patient-level bootstrap intervals.
- Result on test: sensitivity 0.888, specificity 0.853, kappa 0.838. All lower bounds pass the protocol targets on public labels.
- Result on DDR: sensitivity 0.788. The val threshold gives 95% specificity on DDR, so the model scores DDR images lower.
- Cost: about $0.01 (CPU pod).

## 2026-10-15 (afternoon): DDR grade 4 failure analysis

- 121 of 910 DDR grade 4 images are predicted as grade 0, with median probability 0.0003.
- Looked at 64 of them (`scripts/analysis/ddr_grade4_failures.py`). Groups: retina hidden by haze (about 20), large hemorrhage (about 12), laser scars or fibrous tissue (about 10), normal-looking at 128 px (about 19), white-corner crop failure (2).
- Conclusion: the model has no "cannot grade" output, and advanced disease is rare in training.
- Cost: about $0.01 (CPU pod).

## 2026-10-15 (afternoon): White-corner crop bug

- Measured near-white pixels outside the fundus circle in all 57,952 processed images (`scripts/analysis/white_corners.py`).
- Result: 25 images (0.04%) have a white background. ODIR train 12, ODIR test 3, DDR 10. None in EyePACS or APTOS.
- Effect on accuracy: none that we can measure. DDR sensitivity is 0.788 with and without these 10 images. 8 of the 10 DDR images were still referred. The 3 ODIR test images were correct negatives.
- Fixed `crop_fundus` anyway (D-022). The fix applies at the next build.
- First scan measured only the four corners and found 0 images. The padding puts black in the corners, so the scan now measures the whole area outside the circle.
- Cost: about $0.01 (CPU pods).

## 2026-10-15 (evening): Backbone comparison started

- Added `scripts/experiments.txt` and a sweep loop in `pod_job.sh`. Each run gets a smoke run first. A failed run does not stop the others.
- Each backbone now uses the input normalization of its pretrained weights.
- Runs (10 epochs each): EfficientNet-B3, EfficientNet-B5 Noisy Student, ConvNeXt-Small 22k (D-023).
- Pod `csdzc0mp7r3ejv` (RTX 4090, 16 vCPU, $0.89/hr) started 22:24 UTC. The old training pod could not start: its host GPU was taken.

## 2026-10-16 (morning): Backbone comparison result

Runs finished at 01:58 UTC. The pod stopped itself. Total GPU time about 3.5 hours (about $3.10).

| Run | Best epoch | Test AUC | DDR AUC | DDR sensitivity (lower bound) | DDR kappa | DDR VTDR sensitivity | DDR severe undergrades | Minutes per epoch |
|---|---|---|---|---|---|---|---|---|
| Baseline B3, 15 epochs | 9 | 0.939 | 0.914 | 0.788 (0.778) | 0.764 | 0.801 | 193 (16.8%) | 3.5 (8 vCPU) |
| B3, 10 epochs | 4 | 0.956 | 0.969 | 0.877 (0.868) | 0.832 | 0.955 | 72 (6.3%) | 4.4 |
| B5 Noisy Student | 9 | 0.959 | 0.963 | 0.875 (0.866) | 0.800 | 0.912 | 131 (11.4%) | 8.7 |
| ConvNeXt-Small 22k | 9 | 0.957 | 0.947 | 0.869 (0.860) | 0.826 | 0.904 | 105 (9.2%) | 5.4 |

- B3 with 10 epochs is best on DDR. Its best checkpoint is epoch 4. The larger backbones do not help.
- The B3 gain over the baseline is large. The only change is the schedule (10 epochs, best epoch 4 instead of 9). Seed runs check if the gain is real (D-024).
- RETFound access check: the token works (user `alxlin`), but both repositories return 403. The access requests are not approved yet.

## 2026-10-16 (late morning): Seeds, ensemble, and quality head

Seed runs (B3, 10 epochs) on DDR:

| Run | DDR AUC | DDR sensitivity (lower bound) | DDR kappa | DDR VTDR sensitivity | DDR severe undergrades |
|---|---|---|---|---|---|
| Seed 0 | 0.969 | 0.877 (0.868) | 0.832 | 0.955 | 72 |
| Seed 1 | 0.960 | 0.847 (0.837) | 0.828 | 0.891 | 94 |
| Seed 2 | 0.966 | 0.877 (0.867) | 0.810 | 0.907 | 147 |
| Ensemble of 3 | 0.972 | 0.885 (0.876) | 0.843 | 0.937 | 77 (6.7%) |

- All seeds beat the baseline. The ensemble is better than each seed on DDR AUC and kappa. Test AUC 0.962, kappa 0.853.
- Added the quality head (D-025) and `retina-dr ensemble` (D-026). 122 tests pass.
- No RTX 4090 was free in EU-RO-1. The quality run uses an RTX PRO 4000 Blackwell pod (`ojsos1hojydwt0`, $0.57/hr).
- Deleted the old pods `00difbrelhsfon` and `rqcktnogjy5zqf` (approved by Alex).

## 2026-10-16 (afternoon): Review council, quality result, and fixes

- Ran a four-lens review council. Findings and the resulting plan are D-027 to D-031.
- Quality run `b3-e10-q`: EyePACS test quality AUC 0.992. Quality-first rejection failed on DDR (yield 82%, rejection 31% of referable against 7%). Findings-first rule fixes it (yield 92.4%, 8.3% against 7.0%) (D-032).
- The shared quality training lowered DR results (DDR AUC 0.943). The quality model is now a separate run (D-033).
- Added `--task quality`, quality runs in ensembles, run provenance in `config.json`, and pinned EyeQ labels (D-031). 129 tests pass.
- RETFound: both weight files downloaded (access granted). Not used (D-028).
- Pod `6tj7n7p9whuejm` (RTX 4090) trains `quality-b3` and builds `b3-e10-ens3-q`.

## 2026-10-16 (afternoon): Separate quality model and final candidate

Candidate `b3-e10-ens3-q`: the 3 DR seeds plus the separate quality model `quality-b3`, findings-first rule (D-032, D-033). Reject threshold 0.970 from val.

| Split | DR AUC | Yield | Rejected: referable / not referable | Sensitivity / specificity on decided images | VTDR called not referable |
|---|---|---|---|---|---|
| Val | 0.971 | 94.5% | 0.2% / 6.9% | 0.938 / 0.839 | 2 of 192 |
| Test | 0.962 | 94.6% | 1.2% / 6.6% | 0.928 / 0.852 | 0 of 215 |
| DDR (development) | 0.972 | 95.8% | 3.7% / 4.8% | 0.918 / 0.943 | 33 of 1,146 |

- The DR results are the same as the ensemble without quality (D-033 works).
- Quality AUC on EyeQ test rows is 0.992.
- ODIR yield is 85.7% (17.5% of non-referable ODIR images rejected). The quality model treats some ODIR camera images as poor. This shows camera dependence. Set the reject threshold again on clinic tuning data.
- Of the baseline's 121 DDR grade 4 images called grade 0, the candidate refers 67 and the quality model flags 72.
- `quality-b3` config.json has the full commit, manifest SHA-256, and licences (D-031 works).
- Deleted pods `csdzc0mp7r3ejv`, `ae6hziisf38gyo`, `ojsos1hojydwt0`, `6tj7n7p9whuejm` and the CPU readers (approved by Alex).

## 2026-10-16 (afternoon): External test

- Built the external set on the volume: IDRiD 103 images, DeepDRiD 400 images (200 eyes). Manifest SHA-256 4bf899a8b7d15d06...
- IDRiD reader fix: other IDRiD parts and the training split reuse image names, so the reader indexes the test images only.
- Wrote `frozen_model.json` (member commits, best.pt hashes, manifest hash, thresholds, cutpoints).
- Committed the plan hash (`docs/analysis-plan-external.sha256`) before scoring. The scoring pod checked the hash and stored the plan on the volume.
- First scoring attempt stopped at model load: the DR members were trained before the quality head existed, so their checkpoints have no quality-head weights. No prediction was made. Fix: allow only the missing quality-head weights (`load_checkpoint`). This is a loading fix, not a change to the model or the plan.
- Second scoring run finished at 19:47 UTC. Commit 0f7f14a, external manifest SHA-256 4bf899a8... Results are per eye, with the frozen thresholds.

| Set | Eyes (referable) | AUC | Sensitivity (95% interval) | Specificity (95% interval) | System kappa | Yield | Rejected: referable / not | Severe undergrades (system) |
|---|---|---|---|---|---|---|---|---|
| DeepDRiD | 200 (82) | 0.992 | 1.000 (1.000-1.000) | 0.754 (0.651-0.843) | 0.859 | 98.0% | 0% / 3.4% | 0 of 46 |
| IDRiD | 103 (64) | 0.931 | 0.953 (0.895-1.000) | 0.410 (0.271-0.563) | 0.634 | 100% | 0% / 0% | 0 of 32 |

- Verdict under the plan: both sets fail the specificity criterion. IDRiD also fails the kappa criterion. Sensitivity, yield, equal rejection, and severe undergrading pass on both sets.
- Cause: the val threshold (probability 0.040) is too low for these cameras. The ranking is good (AUC 0.992 and 0.931), so a threshold set on data from the target camera can fix this. At 90% specificity, sensitivity is 0.988 (DeepDRiD) and 0.844 (IDRiD). These numbers are diagnostic only; they use the test labels and do not change the plan result.
- IDRiD kappa: the referral head refers 59% of non-referable eyes, and the system rule (D-034) raises those eyes to grade 2. The severity head alone gives kappa 0.775.
- Secondary, DeepDRiD quality: quality AUC 0.809, reject recall 3.6% at the frozen threshold. The quality label mapping is not verified.
- Deviation from the plan: the scoring code reports a 95% interval for the severity-head kappa, not for the system kappa. The system kappa has a point value only.
- Label check: DeepDRiD 82 referable eyes = 164 referable images / 2. IDRiD 64 referable = 32 + 19 + 13 images of grade 2-4. Both agree with the build counts.
- Cost: about $0.40 (GPU and CPU pods). All pods deleted.
