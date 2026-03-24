"""Submission packaging entrypoint for retail track predictions.

Run with::

    uv run --project tracks/retail submit-retail --predictions ./predictions.csv
"""

from __future__ import annotations

import argparse

from vand.retail.submission import prepare_submission, validate_csv


def main() -> None:
    """Validate prediction CSV and create submission zip."""
    parser = argparse.ArgumentParser(description="Retail track submission packaging")
    parser.add_argument("--predictions", type=str, required=True, help="Predictions CSV path")
    parser.add_argument("--output", type=str, default=None, help="Output zip path")
    args = parser.parse_args()

    validate_csv(args.predictions)
    zip_path = prepare_submission(args.predictions, output_path=args.output)

    print(f"Submission package ready: {zip_path}")
    print("Upload this zip file to the Codabench Retail Track submission page.")


if __name__ == "__main__":
    main()
