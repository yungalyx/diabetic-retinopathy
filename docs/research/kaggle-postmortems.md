# Kaggle DR competitions: post-mortem findings

Collected 2026-10-14. Paraphrased from the sources at the end. Items marked "unverified" come from secondary summaries.

## EyePACS 2015 (Diabetic Retinopathy Detection)

Metric: quadratic weighted kappa (QWK). Train 35k, test 53k. Grade 0 is 73% of labels. Grades 3 and 4 are about 2% each.

| Place | Score | Key methods |
|---|---|---|
| 1st, Ben Graham | 0.850 | Rescale to a fixed eye radius. Subtract a blurred local average color ("Ben preprocessing"). Three CNNs, 360° rotation augmentation. A random forest combines both eyes' outputs plus image size into one score, then fixed cutpoints. Color augmentation did not help. |
| 2nd, team o_O | 0.845 | Crop and resize only. 512 px, then 768 px. MSE regression with cutpoints 0.5/1.5/2.5/3.5. Class-balanced sampling decayed toward natural weights. A per-patient blend of both eyes raised kappa from 0.824 to 0.845. |
| 3rd | 0.839 | Unverified. |
| 5th, Jeffrey De Fauw | ~0.84 | 512 px. Kappa-based loss. Expected value of class probabilities with tuned cutpoints. Uniform sampling switched to the natural distribution during training. Pseudo-labels on test images. About a third of the error was true grade 2 predicted as 0. |

## APTOS 2019 (Blindness Detection)

Train 3,662. The leaderboard shook up heavily: public kappa was about 0.80-0.85 and private about 0.93.

| Place | Score | Key methods |
|---|---|---|
| 1st, Guanshuo Xu | 0.936 | SE-ResNeXt and Inception models at 384-512 px. Huber regression, GeM pooling, heavy augmentation. External data: EyePACS 2015, IDRiD, Messidor. Soft pseudo-labels. Details partly unverified. |
| 2nd | 0.934 | EfficientNet-B3 to B5 at 300-460 px with the black border removed. CLAHE and coarse dropout augmentation. Iterative pseudo-labels. |
| 4th | 0.934 | EfficientNet-B4 to B7, lower resolution for larger models. Pretrain on EyePACS 2015, then fine-tune on APTOS. 8x dihedral TTA. |
| 7th/10th, BloodAxe | — | EyePACS data helped slightly. IDRiD and Messidor made scores worse. |

Mid-table teams reported that regression or ordinal targets beat classification. Focal loss, balanced sampling, and direct kappa loss did not help.

## Lessons that held in both competitions

1. Use a regression or ordinal target with tuned cutpoints, not argmax over classes.
2. Use 450-512 px or larger input.
3. Crop the black border. Ben preprocessing is optional.
4. Use heavy rotation, flip, and color augmentation, plus flip or dihedral TTA.
5. Pretrain on EyePACS, then fine-tune on the target data.
6. Combine both eyes of a patient when both are available.

## Data pitfalls

- Label noise: inter-grader kappa is about 0.40-0.65. A replication of Google's DR model on Kaggle EyePACS labels got AUC 0.94, against 0.99 with adjudicated labels.
- APTOS duplicates: about 7% of train images are duplicates. Some duplicate groups have conflicting grades.
- Image size correlates with clinic and grade in APTOS. Size is a shortcut that does not transfer to new data.
- Train and test label distributions differ in APTOS. Some external sets (Messidor, IDRiD) hurt APTOS scores.
- Lens dirt and stripes look like microaneurysms. Some images are ungradable.
- Grade 1 is rare and noisy. Most errors are 0↔1 and 0↔2 confusions.

## Changes applied to this project

- D-006: drop duplicates with conflicting labels.
- D-009: Huber regression with cutpoints tuned on val, never on DDR.
- D-010: augmentation, decayed rebalancing, no image-size features.

## Sources

- https://kaggle-forum-message-attachments.storage.googleapis.com/88655/2795/competitionreport.pdf
- https://github.com/sveitser/kaggle_diabetic
- https://defauw.ai/diabetic-retinopathy-detection/
- https://yerevann.github.io/2015/08/17/diabetic-retinopathy-detection-contest-what-we-did-wrong/
- https://arxiv.org/pdf/1706.04372
- https://speakerdeck.com/hoxomaxwell/kaggle-aptos-2019-at-u-tokyo-med
- https://speakerdeck.com/inoichan/aptos-2019-blindness-detectionfalsematome
- https://github.com/BloodAxe/Kaggle-2019-Blindness-Detection
- https://diyago.github.io/2019/10/04/kaggle-blindness.html
- https://github.com/Fire-Dev2/aptos-leakage-audit
- https://arxiv.org/abs/1803.04337
