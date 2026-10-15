# Progress

Last update: 2026-10-14

## Status

| Step | Status |
|---|---|
| 1. Choose datasets, model design, split plan | Done (D-001 to D-010) |
| 2. Download and build pipeline | Done. Tested on fake data only |
| 3. Research Kaggle post-mortems | Done (docs/research/kaggle-postmortems.md) |
| 4. RunPod access for the agent | Waiting for the user to add the RunPod connector |
| 5. Run download and build on a pod | Not started |
| 6. Training script (EfficientNet-B3, two heads) | Not started |
| 7. RETFound experiment | Not started |

## Next actions

1. User: add the RunPod MCP connector and store `KAGGLE_API_TOKEN` as a RunPod secret.
2. Agent: create a pod (RTX 4090, 250 GB volume) and run the download and build.
3. Agent: verify the items in "Open questions" against the real files.
4. Agent: write the training script.

## Open questions

- Is the DDR Kaggle slug `mariaherrerot/ddrdataset` correct, and does it have `DR_grading/*.txt`?
- Does ODIR `full_df.csv` have the columns the reader expects?
- Does the EyePACS download include `retinopathy_solution.csv` (test labels)?
- Do EyePACS or APTOS images also appear in DDR? The build log reports the count.

## Later improvements

- Near-duplicate detection. Exact dhash misses copies with different compression. Compare hashes by Hamming distance.
- Combine both eyes of one patient (EyePACS, ODIR). This gave about +0.02 kappa in 2015.
- Soft pseudo-labels on unlabeled images.
