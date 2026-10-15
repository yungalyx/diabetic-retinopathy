# Progress

Last update: 2026-10-14

## Status

| Step | Status |
|---|---|
| 1. Choose datasets, model design, split plan | Done (D-001 to D-010) |
| 2. Download and build pipeline | Done. Tested on fake data only |
| 3. Research Kaggle post-mortems | Done (docs/research/kaggle-postmortems.md) |
| 4. RunPod access for the agent | Done. Code is on GitHub (yungalyx/diabetic-retinopathy) |
| 5. Run download and build on a pod | In progress. The pod runs scripts/pod_bootstrap.sh at boot |
| 6. Training script (EfficientNet-B3, two heads) | Not started |
| 7. RETFound experiment | Not started |

## RunPod resources

| Resource | ID | Details | Cost |
|---|---|---|---|
| Pod `retina-dr-4090` | `00difbrelhsfon` | RTX 4090, Secure Cloud, EU-RO-1, template `runpod-torch-v280`, 250 GB container disk | $0.89/hr while running |
| Network volume `retina-dr-data` | `5ac8gl5sy9` | 50 GB STANDARD, EU-RO-1, mounted at `/workspace` | About $3.50/month until deleted |
| Secret `kaggle_api_token` | `s7u36506fd` | Pod reads it as `KAGGLE_API_TOKEN` | Free |

The pod log is also written to `/workspace/bootstrap.log`.

## Next actions

1. Agent: watch the pod log until `/workspace/status/DONE` or `FAILED` exists.
2. Agent: check the build summary table against the expected image counts.
3. Agent: verify the items in "Open questions" against the real files.
4. Agent: write the training script.

## Open questions

- DDR: the slug `mariaherrerot/ddrdataset` is correct. The mirror has one flat folder of 12,524 images and one top-level label file, not grading .txt lists. The reader now accepts a label CSV. Confirm the column names in the pod log.
- ODIR: does `full_df.csv` have the columns the reader expects? Not read yet.
- EyePACS: the download has no test labels (`retinopathy_solution.csv`). Only the 35,126 training images have labels. The 53,576 test images are unused.
- Do EyePACS or APTOS images also appear in DDR? The build log reports the count.

## Later improvements

- Near-duplicate detection. Exact dhash misses copies with different compression. Compare hashes by Hamming distance.
- Combine both eyes of one patient (EyePACS, ODIR). This gave about +0.02 kappa in 2015.
- Soft pseudo-labels on unlabeled images.
