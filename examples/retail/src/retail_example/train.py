# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Training entrypoint for the retail baseline model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from retail_example.model import BaselineModel
from vand.retail import Split
from vand.retail.dataset import get_dataloader


def main() -> None:
    """Train and save the baseline retail model.

    Returns:
        None: Executes the CLI workflow.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./weights")
    parser.add_argument("--use_crops", action="store_true")
    args = parser.parse_args()

    train_loader = get_dataloader(
        root=args.data_root,
        split=Split.TRAIN,
        batch_size=32,
        num_workers=4,
        use_crops=args.use_crops,
    )

    model = BaselineModel()
    model.fit(train_loader, reference_fn=train_loader.dataset.get_reference_images)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model.pt"

    torch.save(model.state_payload(), output_path)
    print(f"Saved model to: {output_path}")


if __name__ == "__main__":
    main()
