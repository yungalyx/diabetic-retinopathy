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
