# Industrial Track — MVTec AD 2

Pixel-level anomaly segmentation on the [MVTec AD 2](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2) dataset.

Official and tested Code Utils for Development are also available for download there if needed.

## What You Need to Do

1. **Implement your model** in `src/industrial/model.py`
2. **Implement training, inference, and submission** in `src/industrial/train.py`, `src/industrial/test.py`, and `src/industrial/submit.py`
3. **Make your final model weights available** in accordance with the official rules to ensure reproducibility and to be eligible for prizes.

## Installation

From the repository root:

```bash
uv sync --all-packages
```

## End-to-End Example

```bash
# 1. Train your model
uv run train-industrial

# 2. Run inference — generate prediction maps in output/
uv run test-industrial

# 3. Validate and package predictions into a submission archive
uv run submit-industrial
```

Upload the resulting `submission.tar.gz` to [benchmark.mvtec.com](https://benchmark.mvtec.com/).

## Dataset

8 categories: `can`, `fabric`, `fruit_jelly`, `rice`, `sheet_metal`, `vial`, `wallplugs`, `walnuts`

Splits: `train`, `validation`, `test_public`, `test_private`, `test_private_mixed`

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

## Evaluation

- **Metric**: SegF1 (pixel-level F1 score)
- **Final rank**: Average rank across `test_private` and `test_private_mixed` splits

> [!NOTE]
> The local evaluation utilities provided in this repository are for convenience and local testing on the public test set of MVTec AD 2 only. Official scores are computed by the challenge server at [benchmark.mvtec.com](https://benchmark.mvtec.com/), whose implementation may differ. No claims or entitlements can be derived from local evaluation results.
