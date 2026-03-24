"""Training entrypoint for the industrial track.

Run with::

    uv run --project tracks/industrial train-industrial --data_root /path/to/mvtec_ad_2
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import json
from pathlib import Path

from industrial.model import AnomalyModel
from vand.industrial import Category, Split
from vand.industrial.dataset import get_dataloader


def main() -> None:
    """Train one model per selected category and save weights + threshold."""
    parser = argparse.ArgumentParser(description="Industrial track training")
    parser.add_argument("--data_root", type=str, required=True, help="MVTec AD 2 dataset root")
    parser.add_argument("--output_dir", type=str, default="./weights", help="Output directory for weights")
    parser.add_argument("--categories", nargs="*", default=None, help="Subset of categories to train")
    args = parser.parse_args()

    categories = args.categories or list(Category)
    output_dir = Path(args.output_dir)

    for category in categories:
        train_loader = get_dataloader(root=args.data_root, category=category, split=Split.TRAIN)
        val_loader = get_dataloader(root=args.data_root, category=category, split=Split.VALIDATION)

        model = AnomalyModel()
        model.fit(train_loader)
        threshold = model.get_threshold(val_loader)

        category_dir = output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        model.save(category_dir / "model.pt")

        threshold_path = category_dir / "threshold.json"
        threshold_path.write_text(json.dumps({"threshold": threshold}, indent=2), encoding="utf-8")

        print(f"[{category}] saved model and threshold={threshold:.6f}")


if __name__ == "__main__":
    main()
