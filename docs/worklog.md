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
