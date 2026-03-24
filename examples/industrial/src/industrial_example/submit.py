# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Submission packaging entrypoint for industrial predictions."""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
from pathlib import Path

from vand.industrial.submission import prepare_submission, validate_submission


def main() -> None:
    """Validate predictions and create a tar.gz submission archive."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="./submission.tar.gz")
    args = parser.parse_args()

    predictions_dir = Path(args.predictions_dir)
    output_path = Path(args.output)

    is_valid = validate_submission(predictions_dir)
    if not is_valid:
        raise RuntimeError("Submission validation returned False")

    archive = prepare_submission(predictions_dir, output_path)
    print(f"Submission ready: {archive}")
    print("Upload this tar.gz to https://benchmark.mvtec.com")


if __name__ == "__main__":
    main()
