# Decision log

Each entry states the decision and the reason. Append new entries. To change a decision, add a new entry that supersedes the old one.

## D-001 Data sources (2026-10-14)

Use EyePACS 2015, APTOS 2019, DDR, and ODIR-5K from Kaggle.
Reason: they are the largest public fundus sets with DR labels, and they come from different countries and cameras.

## D-002 One backbone, two heads (2026-10-14)

Use one image backbone with a binary head and a severity head.
The binary head trains on all training sources. The severity head trains only on rows with a clinician grade. ODIR rows have no severity grade, and their severity loss is masked.
Reason: the backbone gets all the data, and each head learns only from labels it can trust. Two chained models would pass errors from the first to the second.

## D-003 Binary cutoff is grade 2 (2026-10-14)

`binary_label` is 1 for ICDR grade 2 or higher (referable DR).
Reason: this is the standard screening cutoff. Grade 1 has the most grader disagreement.

## D-004 Split plan (2026-10-14)

DDR is fully held out. Other sources split 80/10/10 into train, val, and test by a hash of the patient ID.
Reason: the blended test set measures performance on known cameras. The DDR holdout measures performance on an unseen camera. The gap between the two shows how much the model relies on source-specific signals.

## D-005 Preprocessing (2026-10-14)

Crop to the fundus, pad to a square, resize to 512 px, save as JPEG quality 95.
Reason: top Kaggle solutions used 450-768 px. Gains flatten above about 450 px. Ben Graham preprocessing is an ablation, not the default (see D-010).

## D-006 Deduplication (2026-10-14)

Remove images with the same 64-bit dhash. When duplicates have different labels, drop all copies. When labels agree, keep one copy, and the DDR copy has priority.
Reason: APTOS has about 7% duplicate images, some with conflicting grades. A DDR image in training would leak into the holdout score.

## D-007 Compute (2026-10-14)

Use a RunPod RTX 4090 pod. Use a 250 GB volume during download. Keep only the processed set (about 15 GB) on network storage.
Reason: the model is small. A 4090 trains one 20-epoch run in about 3 hours for about $3.

## D-008 Models (2026-10-14)

Baseline: EfficientNet-B3 (ImageNet weights, `timm`). Second experiment: RETFound.
Choose the model with the better score on the DDR holdout, averaged over 2-3 seeds.
Reason: EfficientNet trains fast and was a top APTOS 2019 backbone. RETFound has retina-specific pretraining.

## D-009 Training targets and metrics (2026-10-14)

Severity head: one continuous output with Huber loss. Convert it to a grade with cutpoints tuned on the val split. Report DDR quadratic weighted kappa with fixed cutpoints (0.5/1.5/2.5/3.5) and with tuned cutpoints.
Binary head: binary cross-entropy. Report AUC and sensitivity at a fixed specificity.
Reason: regression with tuned cutpoints beat classification in both competitions (docs/research/kaggle-postmortems.md).

## D-010 Augmentation and sampling (2026-10-14)

Use full rotation, flips, scale, and color augmentation. Use 8x dihedral test-time augmentation. Start with mild class rebalancing and decay it to the natural distribution during training. Do not copy-oversample grades 3-4.
Do not give the model the original image size or any source ID.
Reason: these methods worked consistently in top solutions. Image size correlates with source and grade, so it is a shortcut that does not transfer.

## D-011 Duplicate detection by fingerprint distance (2026-10-15)

Supersedes the hash part of D-006.
Two images are duplicates when their 32x32 grayscale fingerprints differ by RMS 4 or less (0-255 scale).
Reason: the 64-bit dhash of a 9x8 thumbnail collided for most fundus photos. The first full build dropped 44% of images.
The build logs the nearest-neighbour distance distribution so the threshold can be checked against real data.

## D-012 Duplicate threshold is RMS 2 (2026-10-15)

Supersedes the threshold in D-011 (RMS 4).
Reason: the nearest-neighbour distance distribution on the real data was 0-1: 516, 1-2: 28, 2-4: 684, 4-8: 23,643 images.
Copies sit below 1. The gap is at 1-2. Images at 2-4 are the near edge of the distinct-image distribution.
At RMS 4 the build dropped 877 images, with 111 groups of conflicting labels, which shows false matches.

## D-013 Baseline training setup (2026-10-15)

Model: EfficientNet-B3 (timm, ImageNet weights) with a binary head and a severity head on pooled features, dropout 0.3.
Input: 512 px. Batch 24. bfloat16 autocast on the GPU.
Optimizer: AdamW, learning rate 3e-4, weight decay 1e-5. Linear warmup for 5% of steps, then cosine decay.
Epochs: 15. Sampling: weights proportional to stratum frequency ** -alpha, alpha decays from 0.5 to 0.
Loss: binary cross-entropy plus Huber (delta 1) on rows with a clinician grade, weight 1.
Checkpoint choice: the epoch with the best mean of val AUC and val QWK (cutpoints tuned on val).
Final evaluation: 8x dihedral test-time augmentation. Cutpoints tuned on val only, then applied to test and holdout.
Reason: these follow D-002, D-009, and D-010. The values are common defaults for this model size, not tuned.

## D-014 Augmentation on PIL images, no hue jitter (2026-10-15)

Supersedes the color part of D-010 and D-013.
Run geometric and color augmentation on PIL images, then convert to a tensor.
Use brightness 0.2, contrast 0.2, and saturation 0.1. Do not use hue jitter.
Loader workers default to the cgroup CPU quota minus one, with one thread per worker.
Reason: the first full run trained at 35 images/s with the GPU idle (CPU at 97%).
Tensor augmentation with hue cost 51 ms per image. PIL augmentation without hue costs 14 ms.
os.cpu_count() reported the host CPUs inside the pod, so the loader oversubscribed its 8 vCPUs.

## D-015 Licence status: the current model is research-only (2026-10-15)

DDR images and labels are CC BY-NC-SA 4.0. RETFound weights are CC BY-NC 4.0. EyePACS and APTOS follow Kaggle competition rules. ODIR terms are not recorded.
Treat every model trained in this repo as research-only.
Settle data and weight permissions before training a model for commercial use.
Reason: the researcher's dataset list (docs/research/dataset-licences.md). Public availability does not give commercial permission.

## D-016 DDR is a development holdout. Add an untouched external test (2026-10-15)

Supersedes the holdout role in D-004 and the selection rule in D-008.
DDR stays out of training. Use DDR to compare models and settings.
Use IDRiD Disease Grading and Messidor-2 as the untouched external test. Evaluate them once, on the frozen model only.
Reason: D-008 chooses the model by its DDR score. After that choice, DDR is no longer an independent test.

## D-017 Additional metrics (2026-10-15)

Add these to each evaluation report:
1. Sensitivity and specificity at one operating point. Choose the point on val.
2. Sensitivity for vision-threatening DR (grade 3 or higher).
3. Count of severe undergrading: grade 3-4 eyes predicted as grade 0-1.
4. 95% confidence intervals by bootstrap over patients.
Reason: the project reports accuracy at a fixed operating point with lower confidence bounds, not only AUC.

## D-018 Keep ungradable images for a quality component (2026-10-15)

Keep DDR grade-5 (ungradable) images in all_records.parquet with a quality label. Do not drop them silently.
Report result yield: the fraction of images that get a grade.
DeepDRiD quality labels are the second quality source.
Reason: an accuracy number that excludes failed images hides how often the system gives no result.

## D-019 Record provenance for each source (2026-10-15)

Record the source URL, download date, version, licence text, and file checksums for each release in the build output.
Prefer original releases to Kaggle mirrors. DDR and ODIR come from Kaggle mirrors now.
Reason: labels and image resolution must be traceable to the original release.

## D-020 Continue with research-only data and weights (2026-10-15)

Continue development with the sources and weights in D-015, including RETFound.
The researcher decides how to handle them for a commercial model.
Record the licence of each data source and each pretrained weight in the run config.
Reason: Alex chose to continue and leave the commercial decision to the researcher.

## D-021 Operating point and bootstrap settings (2026-10-15)

Set the operating threshold at 85% specificity on val (all sources). Apply it unchanged to test and holdout.
Use 500 bootstrap resamples of whole patients. Report the 2.5 and 97.5 percentiles.
`retina-dr evaluate` recomputes these metrics for a finished run. It writes `metrics_ci.json` and does not load PyTorch.
Reason: 85% leaves a margin above the 80% specificity target. Images of one patient are not independent.

## D-022 White background counts as background in the crop (2026-10-15)

`crop_fundus` treats pixels with all channels at 235 or more as background, as it treats near-black pixels.
After the crop, white pixels outside the inscribed ellipse become black.
Reason: some source images have a white background. The old crop kept it, so the fundus was smaller and the corners were white.
The fix takes effect at the next build. The current processed set still has the old crops.

## D-023 Backbone comparison with a 10-epoch schedule (2026-10-15)

Train three backbones on the same manifest and splits, 10 epochs each: EfficientNet-B3, EfficientNet-B5 (Noisy Student, JFT), and ConvNeXt-Small (ImageNet-22k).
List the runs in `scripts/experiments.txt`. Compare them on the DDR holdout.
Use the input normalization from the pretrained config of each backbone.
Reason: the baseline peaked at epoch 9. The B5 and ConvNeXt weights are Apache 2.0, unlike the RETFound weights (CC BY-NC 4.0). RETFound needs gated access on HuggingFace.

## D-024 EfficientNet-B3 with 10 epochs is the current best model (2026-10-16)

Result of D-023 on the DDR holdout: B3 AUC 0.969, B5 0.963, ConvNeXt-Small 0.947. B3 also has the best DDR kappa (0.832) and the fewest severe undergrades (72).
The larger backbones do not improve DDR results and train 1.2 to 2 times slower.
Train B3 with seeds 1 and 2 before the choice is final. The baseline (15 epochs) gave DDR AUC 0.914, so the run-to-run spread can be large.
Reason: D-008 requires 2-3 seeds per model choice.

## D-025 Image quality head trained on EyeQ labels (2026-10-16)

Add a third head that gives the probability that an image cannot be graded ("reject").
Train it on the EyeQ quality labels (Good, Usable, Reject). EyeQ covers 12,543 EyePACS images in our manifest. The labels are CC BY-NC-SA 4.0 (research-only, as D-015).
Do not train the grading heads on EyeQ Reject images. A grade read from an ungradable image is not reliable.
Set the reject threshold on val only: the Youden threshold on EyeQ-labelled rows, raised when needed so that at least 90% of val images get a result. Apply it unchanged to test and DDR.
Report yield, rejection rate in positive and negative eyes, reject recall on EyeQ rows, grading metrics on accepted images, and the number of severe undergrades that the quality head rejects.
Reason: the DDR failure analysis showed that the model calls images with a hidden retina grade 0. The study protocol requires 90% DR yield and limits on acceptance of inadequate images.
The quality head learns only from human quality labels. It cannot learn to reject hard disease cases to improve the grading loss.
The DDR Kaggle mirror has no ungradable images (12,522 of 13,673), so DDR cannot test reject recall. DeepDRiD quality labels can do this later.

## D-026 Seed result and seed ensembles (2026-10-16)

B3 with 10 epochs, three seeds, on DDR: AUC 0.969 / 0.960 / 0.966, kappa 0.832 / 0.828 / 0.810, severe undergrades 72 / 94 / 147.
All three seeds beat the 15-epoch baseline (AUC 0.914). The gain is real. Keep B3 with 10 epochs.
Severe undergrading changes much between seeds. Average the seeds into one ensemble and compare it as one model.
`retina-dr ensemble` averages the logits, severity values, and reject probabilities of finished runs. It tunes cutpoints and thresholds on val only.
List ensembles in `scripts/ensembles.txt`. `pod_job.sh` builds them after the training runs.
Reason: averaging seeds usually lowers variance. It costs no training, only 3 times the inference.

## D-027 Review council findings (2026-10-16)

Four independent reviews (clinical and study fit, ML methodology, data governance, project lead) read the docs and code. The full text is in `docs/private/council-2026-10-16.md` (not tracked).
Findings that all reviews agree on:
1. DDR is spent. We used it to choose the backbone, the schedule, the seeds, and the ensemble. DDR results are now optimistic, and the 10-epoch choice broke the rule in CLAUDE.md.
2. All metrics are per image. The study counts eyes. There is no rule that joins images of one eye, and the referral output (binary head) and the severity grade can disagree for one eye.
3. DDR and APTOS have no patient IDs. Their bootstrap intervals are too narrow, and APTOS test results can contain both eyes of one patient in different splits.
4. The val threshold does not transfer across cameras (85% val specificity gave about 95% on DDR). Thresholds must be set again on clinic tuning data.
5. Provenance is incomplete. Run configs have no commit or manifest hash (D-019 is not done). Every run used the set built before D-022. EyeQ labels are not pinned and stay on pod-local disk.
6. Licences: EyeQ (CC BY-NC-SA) now trains the quality head, and RETFound is CC BY-NC. The exact EyePACS and APTOS rule text and the ODIR terms are not on file.
7. The critical path is outside this repo: clinic data access and the macular edema (OCT) component.

## D-028 Freeze the public-data DR configuration (2026-10-16)

Supersedes the step 7 plan in D-008.
The DR configuration is EfficientNet-B3, 10 epochs, ensemble of seeds 0-2, plus the quality head (D-025).
Do not use DDR to choose anything more. Report DDR as a development result only.
Stop: the RETFound comparison, more backbone or seed sweeps, both-eye fusion, pseudo-labels.
Reason: D-027 findings 1 and 7. More public-data tuning does not move the study gate, and DDR cannot judge it without bias.

## D-029 External test with a written plan, run once (2026-10-16)

Write `docs/analysis-plan-external.md` before any IDRiD or Messidor-2 image is scored. Commit it first.
The plan fixes: eye as the unit, the image-to-eye rule, one referral rule, the threshold and cutpoints (from val), rejected eyes counted against yield, a severe-undergrading limit, and the pass criteria.
Run the frozen D-028 model on IDRiD (Disease Grading test split) and Messidor-2 (adjudicated grades) once. Report every result, pass or fail.
Reason: these are the only clean public test sets left (D-027 finding 1). Messidor-2 grades are adjudicated, which is closest to the study reference standard.

## D-030 Per-eye results and one referral rule (2026-10-16)

Add an eye key (patient and laterality) where the source gives it. Where it does not (APTOS, DDR), treat each image as one eye and state this limit in every report.
Eye result: the mean of the image outputs of that eye, after quality rejection of single images. An eye with no accepted image gets "no result".
Referral rule: refer when the binary probability is above the threshold OR the severity grade is 2 or higher. Report how often the two heads disagree.
Report eye-level yield, sensitivity, specificity, and kappa, with intervals clustered by patient.
Reason: D-027 findings 2 and 3. The study counts eyes and needs one locked decision rule.

## D-031 Provenance in every build and run (2026-10-16)

Implements D-019.
Each run writes the full git commit, the SHA-256 of the manifest it read, and the licence of each data source and weight into `config.json`.
Pin EyeQ to a fixed commit of its repository and check its SHA-256. Store the labelled manifest on the network volume.
Rebuild the processed set at a tagged commit before the external test, so the frozen model uses the D-022 crop.
Reason: D-027 finding 5. The external result must trace to one commit and one data version.

## D-032 Findings-first decision rule (2026-10-16)

Result of the quality run `b3-e10-q` (single seed), with the reject threshold from val (0.980):
- EyePACS test: quality AUC 0.992, reject recall 0.72.
- With "reject first" (rule A), DDR yield is 82% and the quality head rejects 31% of referable images against 7% of non-referable images. This fails the yield target and the equal-rejection requirement.
- Of the baseline's 121 DDR grade 4 images called grade 0, this model rejects 75 and refers 46. None is called "no DR".
Use "findings first" (rule B): refer when the referral probability is above its threshold, whatever the image quality. Reject only an image that would otherwise be called "not referable".
Rule B on DDR: yield 92.4%, rejection 8.3% of referable against 7.0% of non-referable, sensitivity 0.916 and specificity 0.945 on decided images. Test: yield 94.1%, rejection 2.0% against 7.0%.
Reason: a referable finding on a poor image is still a referral. The study sends inadequate images to clinical examination, so "no result" is safe only for images the model would clear.
Note: DDR showed the problem, but the rule follows from the clinical pathway, not from a DDR score. Thresholds stay from val.

## D-033 Train the quality head as a separate model (2026-10-16)

The quality run lowered DR results on DDR: AUC 0.943 against 0.960-0.969 for the three DR seeds, kappa 0.801, severe undergrades 143.
Keep the frozen DR ensemble (D-028) unchanged. Train the quality model separately: train rows with an EyeQ label only, quality loss only, checkpoint by val quality AUC.
The ensemble takes referral and severity from the DR members and the reject probability from the quality model.
Reason: the shared training (quality loss and reject masking) changed the DR heads. A separate model keeps the DR result that we already measured.

## D-034 One decision rule: the referral head decides, the grade follows (2026-10-16)

Supersedes the "OR" referral rule in D-030.
Refer when the referral probability is above the val threshold. A referred eye gets grade max(severity grade, 2). An eye that is not referred gets grade min(severity grade, 1).
Report `head_disagreement` (how often the two heads disagree before this rule), `qwk_system`, and `severe_undergraded_system` from the system grades.
In every current public source, one image is one eye (EyePACS and ODIR have one image per eye; APTOS, DDR, IDRiD, and Messidor-2 have no eye link). `to_eye_level` averages the images of one eye for later clinic data.
Reason: the "OR" rule changes the operating point that val fixed. This rule keeps the val threshold and gives one consistent output per eye.

## D-035 External sets, frozen model record, and no rebuild (2026-10-16)

External sets for the one-time test (D-029):
- IDRiD Disease Grading test split (103 images). Kaggle copy `aaryapatel98/indian-diabetic-retinopathy-image-dataset`. CC BY 4.0 permits copies.
- DeepDRiD regular fundus, Online-Challenge 1 and 2 evaluation set (400 images, 200 eyes, 100 patients), from the official GitHub repository. CC BY-SA 4.0, used for evaluation only. It also has image-quality labels ("Overall quality": 1 good, 0 not acceptable), so it is the external test of the quality model.
- Messidor-2: images only from ADCIS (no copies: ADCIS forbids redistribution), grades from the Google Brain Kaggle release. Waits for ADCIS access. It runs later with the same plan.
Supersedes the rebuild in D-031. Do not rebuild or retrain before the external test. The frozen model is the D-028 ensemble `b3-e10-ens3-q`, recorded in `runs/exp/b3-e10-ens3-q/frozen_model.json` (member commits, best.pt SHA-256, manifest SHA-256, thresholds, cutpoints).
Reason: the D-022 crop fix changes 25 of 57,952 training images and had no measurable effect. A retrain makes a new model that DDR cannot judge without bias. External images use the fixed crop, so their backgrounds match the other training images.
The analysis plan is in `docs/private/analysis-plan-external.md` because it uses unpublished study criteria. Its SHA-256 is committed in `docs/analysis-plan-external.sha256` before any external image is scored.

## D-036 External test result and what follows (2026-10-16)

The frozen model (D-035) passed sensitivity, yield, equal rejection, and severe undergrading on IDRiD and DeepDRiD. It failed specificity on both (0.754 and 0.410 at the frozen threshold) and system kappa on IDRiD (0.634). See the worklog for the full table.
The AUC is high (0.992 and 0.931). The failure comes from the threshold, which was set on val images from other cameras.
Decisions:
1. Report this result as it is. Do not change the model or the threshold and score these sets again. They are now spent, like DDR.
2. The clinic phase must set the referral and reject thresholds on clinic tuning data before its frozen test. This result is the evidence for that step.
3. Messidor-2 runs later with the same plan and the same frozen model, when ADCIS access exists.
4. Add a 95% interval for the system kappa to the scoring code before the next external or clinic test.
