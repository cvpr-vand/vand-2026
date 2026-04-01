# Retail Track — Kaputt 2

Image-level defect detection on the [Kaputt 2](https://www.kaputt-dataset.com/kaputt2) dataset.

## What You Need to Do

1. **Implement your model** in `src/retail/model.py`
2. **Implement training and inference** in `src/retail/train.py` and `src/retail/test.py`

## Installation

From the repository root:

```bash
uv sync --all-packages
```

## End-to-End Example

```bash
# 1. Train your model
uv run train-retail

# 2. Run inference — generate a predictions CSV
uv run test-retail

# 3. Package the CSV into a submission zip
uv run submit-retail package predictions.csv

# 4. Validate the zip before uploading
uv run submit-retail validate predictions.zip --ground-truth query-test.parquet
```

Upload `predictions.zip` to [Codabench](https://www.codabench.org/).

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

The `submit-retail package` command validates the CSV and zips it.
The `submit-retail validate` command checks an existing zip against the ground
truth parquet to verify schema, value ranges, and full coverage before uploading.

> [!NOTE]
> The local evaluation utilities provided in this repository are for convenience and local testing only. Official scores are computed by the challenge server on [Codabench](https://www.codabench.org/), whose implementation may differ. No claims or entitlements can be derived from local evaluation results.
