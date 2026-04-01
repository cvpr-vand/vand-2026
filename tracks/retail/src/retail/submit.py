"""Submission packaging entrypoint for retail track predictions."""

import argparse
import sys

from retail.submission import prepare_submission, validate_submission_zip

QUERY_TEST_PARQUET_PATH = "query-test.parquet"


def main() -> None:
    """Validate prediction CSV and create submission zip.

    Save your results as csv and package it in a zip.
    Upload this zip file to the Codabench Retail Track submission page.
    """
    parser = argparse.ArgumentParser(
        description="Validate and package a Kaputt 2 retail track submission."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- package command -------------------------------------------------------
    pkg = subparsers.add_parser(
        "package",
        help="Validate a prediction CSV and package it into a submission zip.",
    )
    pkg.add_argument("csv_path", help="Path to the prediction CSV file.")
    pkg.add_argument(
        "--output",
        default=None,
        help="Optional output zip path (default: same name as CSV with .zip).",
    )

    # -- validate command ------------------------------------------------------
    val = subparsers.add_parser(
        "validate",
        help="Validate an existing submission zip file.",
    )
    val.add_argument("submission_zip", help="Path to the submission zip file.")
    val.add_argument(
        "--ground-truth",
        default=QUERY_TEST_PARQUET_PATH,
        help=f"Path to the ground truth parquet file (default: {QUERY_TEST_PARQUET_PATH}).",
    )

    args = parser.parse_args()

    if args.command == "package":
        try:
            zip_path = prepare_submission(args.csv_path, args.output)
            print(f"Submission packaged: {zip_path}")
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "validate":
        try:
            validate_submission_zip(args.submission_zip, args.ground_truth)
            print("Submission looks good.")
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
