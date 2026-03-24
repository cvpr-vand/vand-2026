"""Submission packaging entrypoint for industrial predictions.

Run with::

    uv run --project tracks/industrial submit-industrial --predictions_dir ./predictions
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
from pathlib import Path

from vand.industrial.submission import prepare_submission, validate_submission


def main() -> None:
    """Validate predictions and create a tar.gz submission archive."""
    parser = argparse.ArgumentParser(description="Industrial track submission packaging")
    parser.add_argument("--predictions_dir", type=str, required=True, help="Predictions directory")
    parser.add_argument("--output", type=str, default="./submission.tar.gz", help="Output archive path")
    args = parser.parse_args()

    predictions_dir = Path(args.predictions_dir)
    output_path = Path(args.output)

    validate_submission(predictions_dir)
    archive = prepare_submission(predictions_dir, output_path)
    print(f"Submission ready: {archive}")
    print("Upload this tar.gz to https://benchmark.mvtec.com")


if __name__ == "__main__":
    main()
