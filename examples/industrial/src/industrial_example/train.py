# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Training entrypoint for industrial track baseline models."""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def main() -> None:
    """Train one baseline model and threshold per selected category."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./weights")
    parser.add_argument("--categories", nargs="*", default=None)
    args = parser.parse_args()

    categories = _parse_categories(args.categories)
    output_dir = Path(args.output_dir)

    for category in categories:
        train_loader = get_dataloader(
            root=args.data_root,
            category=category,
            split=Split.TRAIN,
            batch_size=16,
            num_workers=4,
        )
        val_loader = get_dataloader(
            root=args.data_root,
            category=category,
            split=Split.VALIDATION,
            batch_size=16,
            num_workers=4,
        )

        model = BaselineModel()
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
