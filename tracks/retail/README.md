# Retail Track — Kaputt 2

Image-level defect detection on the [Kaputt 2](https://www.kaputt-dataset.com/kaputt2) dataset.

## What You Need to Do

1. **Implement your model** in `src/retail/model.py`
2. Fill in all methods of `DefectModel` (currently raise `NotImplementedError`)
3. Use the wired entrypoints to train, generate predictions, and package your submission

See `examples/retail/` for a working baseline implementation.

## Model Interface

Your `DefectModel` must implement:

| Method                                          | Purpose                             |
| ----------------------------------------------- | ----------------------------------- |
| `fit(train_dataloader, reference_fn)`           | Train on Kaputt (labeled) data      |
| `predict(image, reference_embeddings) → float`  | Single-image defect score in [0, 1] |
| `predict_batch(images) → Tensor`                | Batch defect scores                 |
| `state_payload() → dict` / `load_payload(dict)` | Checkpoint serialization            |

## Dataset

- **Training**: Kaputt (labeled — defect severity, defect types, materials)
- **Testing**: Kaputt 2 (test-only, no labels)
- Each query image has 1–3 reference images of the same item

## Workflow

```bash
# 1. Train on Kaputt
uv run --project tracks/retail train-retail \
    --data_root /path/to/kaputt

# 2. Generate predictions on Kaputt 2
uv run --project tracks/retail test-retail \
    --data_root /path/to/kaputt2 \
    --weights_path ./weights/model.pt

# 3. Package submission
uv run --project tracks/retail submit-retail \
    --predictions ./predictions.csv
```

Use `--use_crops` to train/test on cropped images instead of full images.

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

## Evaluation

- **Primary metric**: AP_any (Average Precision — any defect vs no defect)
- **Auxiliary metrics**: AP_major, AUROC, R@50%P, R@1%FPR
- **Limit**: 2 submissions per day

> [!NOTE]
> The local evaluation utilities provided in this repository are for convenience and local testing only. Official scores are computed by the challenge server on [Codabench](https://www.codabench.org/), whose implementation may differ. No claims or entitlements can be derived from local evaluation results.

## Rules

- Data usage restrictions (Kaputt 2 cannot be used for training)
- Zero-Shot/Few-Shot category constraints
- Code and report submission requirements

> [!CAUITION]
> ADD RULES
