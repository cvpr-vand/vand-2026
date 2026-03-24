"""Inference entrypoint that writes submission-ready industrial predictions.

Run with::

    uv run --project tracks/industrial test-industrial \\
        --data_root /path/to/mvtec_ad_2 --weights_dir ./weights
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from PIL import Image

from industrial.model import AnomalyModel
from vand.industrial import Category, Split
from vand.industrial.dataset import get_dataloader


def _save_predictions(
    model: AnomalyModel,
    dataloader: Any,
    anom_dir: Path,
    thresh_dir: Path,
    split: Split,
    threshold: float,
) -> None:
    """Run inference and save TIFF anomaly maps and thresholded PNG masks."""
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
            tifffile.imwrite(anom_dir / f"{base_name}.tiff", arr)
            binary = np.where(arr > threshold, 255, 0).astype(np.uint8)
            Image.fromarray(binary, mode="L").save(thresh_dir / f"{base_name}.png")
            index += 1


def main() -> None:
    """Generate predictions for private splits across selected categories."""
    parser = argparse.ArgumentParser(description="Industrial track inference")
    parser.add_argument("--data_root", type=str, required=True, help="MVTec AD 2 dataset root")
    parser.add_argument("--weights_dir", type=str, required=True, help="Directory with trained weights")
    parser.add_argument("--output_dir", type=str, default="./predictions", help="Output directory for predictions")
    parser.add_argument("--categories", nargs="*", default=None, help="Subset of categories")
    args = parser.parse_args()

    categories = args.categories or list(Category)
    weights_dir = Path(args.weights_dir)
    output_dir = Path(args.output_dir)

    for category in categories:
        model_path = weights_dir / category / "model.pt"
        threshold_path = weights_dir / category / "threshold.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing weights: {model_path}")
        if not threshold_path.exists():
            raise FileNotFoundError(f"Missing threshold file: {threshold_path}")

        threshold = float(json.loads(threshold_path.read_text(encoding="utf-8"))["threshold"])
        model = AnomalyModel.load(model_path)

        for split in (Split.TEST_PRIVATE, Split.TEST_PRIVATE_MIXED):
            loader = get_dataloader(root=args.data_root, category=category, split=split, batch_size=4)
            _save_predictions(
                model=model,
                dataloader=loader,
                anom_dir=output_dir / "anomaly_images" / category / split,
                thresh_dir=output_dir / "anomaly_images_thresholded" / category / split,
                split=split,
                threshold=threshold,
            )

        print(f"[{category}] predictions written")


if __name__ == "__main__":
    main()
