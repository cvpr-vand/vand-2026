# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Inference entrypoint that writes submission-ready industrial predictions."""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from PIL import Image

from industrial_example.model import BaselineModel
from vand.industrial import Category, Split
from vand.industrial.dataset import get_dataloader


def _parse_categories(raw: list[str] | None) -> list[str]:
    """Validate and normalize optional category arguments."""

    categories = list(Category)
    if raw is None or len(raw) == 0:
        return categories
    invalid = [category for category in raw if category not in categories]
    if len(invalid) > 0:
        raise ValueError(f"Invalid categories: {invalid}")
    return raw


def _save_predictions(
    model: BaselineModel,
    dataloader: Any,
    anom_dir: Path,
    thresh_dir: Path,
    split: Split,
    threshold: float,
) -> None:
    """Run inference and save TIFF anomaly maps plus thresholded PNG masks."""

    suffix = "regular" if split == Split.TEST_PRIVATE else "mixed"
    anom_dir.mkdir(parents=True, exist_ok=True)
    thresh_dir.mkdir(parents=True, exist_ok=True)

    index = 1
    for batch in dataloader:
        images = batch.image
        batch_size = int(images.shape[0])
        for i in range(batch_size):
            anomaly_map, _ = model.predict(images[i])
            arr = anomaly_map.numpy().astype(np.float16)
            base_name = f"{index:03d}_{suffix}"
            tiff_path = anom_dir / f"{base_name}.tiff"
            png_path = thresh_dir / f"{base_name}.png"
            tifffile.imwrite(tiff_path, arr)
            binary = np.where(arr > threshold, 255, 0).astype(np.uint8)
            Image.fromarray(binary, mode="L").save(png_path)
            index += 1


def main() -> None:
    """Generate predictions for private splits across selected categories."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--weights_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./predictions")
    parser.add_argument("--categories", nargs="*", default=None)
    args = parser.parse_args()

    categories = _parse_categories(args.categories)
    weights_dir = Path(args.weights_dir)
    output_dir = Path(args.output_dir)

    for category in categories:
        category_weight_dir = weights_dir / category
        model_path = category_weight_dir / "model.pt"
        threshold_path = category_weight_dir / "threshold.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing weights: {model_path}")
        if not threshold_path.exists():
            raise FileNotFoundError(f"Missing threshold file: {threshold_path}")

        threshold_json = json.loads(threshold_path.read_text(encoding="utf-8"))
        threshold = float(threshold_json["threshold"])
        model = BaselineModel.load(model_path)

        for split in (Split.TEST_PRIVATE, Split.TEST_PRIVATE_MIXED):
            loader = get_dataloader(
                root=args.data_root,
                category=category,
                split=split,
                batch_size=4,
                num_workers=4,
            )
            target_dir = output_dir / "anomaly_images" / category / split
            target_thresh_dir = output_dir / "anomaly_images_thresholded" / category / split
            target_dir.mkdir(parents=True, exist_ok=True)
            target_thresh_dir.mkdir(parents=True, exist_ok=True)
            _save_predictions(
                model=model,
                dataloader=loader,
                anom_dir=target_dir,
                thresh_dir=target_thresh_dir,
                split=split,
                threshold=threshold,
            )

        print(f"[{category}] predictions written")


if __name__ == "__main__":
    main()
