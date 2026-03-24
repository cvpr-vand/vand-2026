# Retail Track — Kaputt 2

Image-level defect detection on the [Kaputt 2](https://www.kaputt-dataset.com/kaputt2) dataset.

## What You Need to Do

1. **Implement your model** in `src/retail/model.py`
2. **Implement training, inference, and submission** in `src/retail/train.py`, `src/retail/test.py`, and `src/retail/submit.py`

## Submission Format

CSV file with two columns:

```csv
capture_id,pred
img_00001,0.85
img_00002,0.12
img_00003,0.94
```

- `capture_id`: Image identifier from the Kaputt 2 dataset
- `pred`: Defect score between 0 and 1

The `submit` command validates and zips the CSV. Upload to [Codabench](https://www.codabench.org/).


> [!NOTE]
> The local evaluation utilities provided in this repository are for convenience and local testing only. Official scores are computed by the challenge server on [Codabench](https://www.codabench.org/), whose implementation may differ. No claims or entitlements can be derived from local evaluation results.
