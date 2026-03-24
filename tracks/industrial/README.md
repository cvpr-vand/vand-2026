# Industrial Track — MVTec AD 2

Pixel-level anomaly segmentation on the [MVTec AD 2](https://www.mvtec.com/company/research/datasets/mvtec-ad-2) dataset.

## What You Need to Do

1. **Implement your model** in `src/industrial/model.py`
2. Fill in all methods of `AnomalyModel` (currently raise `NotImplementedError`)
3. Use the wired entrypoints to train, generate predictions, and package your submission

See `examples/industrial/` for a working baseline implementation.

## Model Interface

Your `AnomalyModel` must implement:

| Method                                  | Purpose                                     |
| --------------------------------------- | ------------------------------------------- |
| `fit(train_dataloader)`                 | Train on normal images                      |
| `predict(image) → (anomaly_map, score)` | Pixel-level anomaly map + scalar score      |
| `get_threshold(val_dataloader) → float` | Binarization threshold from validation data |
| `save(path)` / `load(path)`             | Checkpoint serialization                    |

## Dataset

8 categories: `can`, `fabric`, `fruit_jelly`, `rice`, `sheet_metal`, `vial`, `wallplugs`, `walnuts`

Splits: `train`, `validation`, `test_public`, `test_private`, `test_private_mixed`

## Workflow

```bash
# 1. Train (one model per category)
uv run --project tracks/industrial train-industrial \
    --data_root /path/to/mvtec_ad_2

# 2. Generate predictions (TIFF anomaly maps + thresholded PNGs)
uv run --project tracks/industrial test-industrial \
    --data_root /path/to/mvtec_ad_2 \
    --weights_dir ./weights

# 3. Package submission
uv run --project tracks/industrial submit-industrial \
    --predictions_dir ./predictions
```

You can train/test on a subset of categories:

```bash
uv run --project tracks/industrial train-industrial \
    --data_root /path/to/mvtec_ad_2 \
    --categories can fabric
```

## Submission Format

The `submit` command creates a `submission.tar.gz` containing:

```
anomaly_images/
  {category}/
    test_private/
      001_regular.tiff    # float16 anomaly maps
      ...
    test_private_mixed/
      001_mixed.tiff
      ...
anomaly_images_thresholded/
  {category}/
    test_private/
      001_regular.png     # binary masks {0, 255}
      ...
    test_private_mixed/
      001_mixed.png
      ...
```

Upload to [benchmark.mvtec.com](https://benchmark.mvtec.com/).

## Evaluation

- **Metric**: SegF1 (pixel-level F1 score)
- **Final rank**: Average rank across `test_private` and `test_private_mixed` splits
- **Limit**: 3 submissions per week

> [!NOTE]
> The local evaluation utilities provided in this repository are for convenience and local testing only. Official scores are computed by the challenge server at [benchmark.mvtec.com](https://benchmark.mvtec.com/), whose implementation may differ. No claims or entitlements can be derived from local evaluation results.

## Rules

- Full-Shot vs Zero-Shot settings
- Model design constraints (class-agnostic requirement)
- Code and report submission requirements

> [!CAUITION]
> ADD RULES
