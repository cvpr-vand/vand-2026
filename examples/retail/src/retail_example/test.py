# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Inference entrypoint for retail baseline predictions."""

from __future__ import annotations

import argparse

import torch

from retail_example.model import BaselineModel
from vand.retail import Split
from vand.retail.dataset import get_inference_dataloader
from vand.retail.submission import generate_csv


def main() -> None:
    """Run inference and write predictions CSV.

    Returns:
        None: Executes the CLI workflow.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="./predictions.csv")
    parser.add_argument("--use_crops", action="store_true")
    args = parser.parse_args()

    dataloader = get_inference_dataloader(
        root=args.data_root,
        split=Split.TEST,
        use_crops=args.use_crops,
    )

    model = BaselineModel()
    payload = torch.load(args.weights_path, map_location=model.device, weights_only=True)
    model.load_payload(payload)
    model.eval()

    predictions: dict[str, float] = {}
    with torch.no_grad():
        for batch in dataloader:
            scores = model.predict_batch(batch.image)
            capture_ids = batch.capture_id
            for capture_id, score in zip(capture_ids, scores.tolist(), strict=False):
                predictions[str(capture_id)] = float(score)

    output = generate_csv(predictions, output_path=args.output)
    print(f"Saved predictions to: {output}")


if __name__ == "__main__":
    main()
