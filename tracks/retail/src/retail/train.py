"""Training entrypoint for the retail track.

Run with::

    uv run --project tracks/retail train-retail --data_root /path/to/kaputt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from retail.model import DefectModel
from vand.retail import Split
from vand.retail.dataset import get_dataloader


def main() -> None:
    """Train the defect model and save weights."""
    parser = argparse.ArgumentParser(description="Retail track training")
    parser.add_argument("--data_root", type=str, required=True, help="Kaputt dataset root")
    parser.add_argument("--output_dir", type=str, default="./weights", help="Output directory for weights")
    parser.add_argument("--use_crops", action="store_true", help="Use cropped images instead of full images")
    args = parser.parse_args()

    train_loader = get_dataloader(
        root=args.data_root,
        split=Split.TRAIN,
        batch_size=32,
        num_workers=4,
        use_crops=args.use_crops,
    )

    model = DefectModel()
    model.fit(train_loader, reference_fn=train_loader.dataset.get_reference_images)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model.pt"

    torch.save(model.state_payload(), output_path)
    print(f"Saved model to: {output_path}")


if __name__ == "__main__":
    main()
