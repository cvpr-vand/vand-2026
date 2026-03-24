# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Submission packaging entrypoint for retail track predictions."""

from __future__ import annotations

import argparse

from vand.retail.submission import prepare_submission, validate_csv


def main() -> None:
    """Validate prediction CSV and create submission zip.

    Returns:
        None: Executes the CLI workflow.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    validate_csv(args.predictions)
    zip_path = prepare_submission(args.predictions, output_path=args.output)

    print(f"Submission package ready: {zip_path}")
    print("Upload this zip file to the Codabench Retail Track submission page.")


if __name__ == "__main__":
    main()
