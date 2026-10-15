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
