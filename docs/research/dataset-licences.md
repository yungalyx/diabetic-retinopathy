# Dataset and weight licences

Collected 2026-10-15 from a dataset list given by the project researcher (source pages checked 2026-10-14). RETFound licence checked 2026-10-15 in its repository LICENSE file.

## Summary

| Source | Use in this repo | Licence or terms | Commercial use |
|---|---|---|---|
| EyePACS 2015 | Train, val, test | Kaggle competition rules | Not established. Confirm permission first |
| APTOS 2019 | Train, val, test | Kaggle competition rules | Not established. Confirm permission first |
| ODIR-5K | Train, val, test (binary only) | Not recorded. Kaggle mirror | Unknown. Find the original release and its terms |
| DDR | Holdout | Images and labels: CC BY-NC-SA 4.0. The MIT licence covers only the code | No, without written permission from the authors |
| IDRiD (Disease Grading) | Not used yet | CC BY 4.0 | Yes, with attribution |
| DeepDRiD | Not used yet | CC BY-SA 4.0 | Yes, with attribution and share-alike. Review share-alike terms for model weights |
| Messidor-2 | Not used yet | ADCIS: free research and education use. No redistribution. DR grades are a separate Google Brain release on Kaggle | No, without authorization |
| RETFound weights | Planned (step 7) | CC BY-NC 4.0 | No |
| ImageNet weights (timm) | Baseline backbone | Not recorded | Not reviewed |

Result: the current model and any RETFound model are research-only. A commercial model needs permissions settled before training.

## Source links

| Source | Download | Terms |
|---|---|---|
| EyePACS | https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data | https://www.kaggle.com/competitions/diabetic-retinopathy-detection/rules |
| APTOS 2019 | https://www.kaggle.com/competitions/aptos2019-blindness-detection/data | https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules |
| DDR | https://github.com/nkicsl/DDR-dataset | https://github.com/nkicsl/DDR-dataset#license |
| IDRiD | https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid | https://idrid.grand-challenge.org/Data/ |
| DeepDRiD | https://github.com/deepdrdoc/DeepDRiD | https://github.com/deepdrdoc/DeepDRiD/blob/master/LICENSE |
| Messidor-2 images | https://www.adcis.net/en/third-party/messidor2/ | Same page |
| Messidor-2 grades | https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades | Same page |
| RETFound | https://github.com/rmaphoh/RETFound | Repository LICENSE file |

## Notes per source

- IDRiD: use the Disease Grading part (516 images). Keep its given train and test split when reporting benchmark results. Its macular edema labels come from fundus photos, not OCT.
- DeepDRiD: about 2,000 regular fundus images with DR grades and image-quality labels, plus an ultra-widefield set. Evaluate the two image types separately.
- Messidor-2: 1,748 images from 874 examinations. The image release has no DR labels. Match images to the Kaggle grades and record the source of both.
- DDR: about 13,700 images, including an ungradable class. The build drops ungradable images (grade 5) now.

## Data handling rules from the researcher

1. Prefer the original release to a Kaggle re-upload. Our DDR and ODIR copies are Kaggle mirrors.
2. Record the source URL, download date, version, licence text, and file checksums for each release.
3. Document how each source's labels map to ICDR grades 0-4.
4. Keep all images of one patient in one split. Document sources without patient IDs (APTOS, DDR).
5. Keep one test set untouched. Do not tune a model on the set that reports its independent result.
