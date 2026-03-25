# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Submission file generation and validation utilities."""

import csv
import zipfile
from pathlib import Path


def generate_csv(
    predictions: dict[str, float], output_path: str | Path = "predictions.csv"
) -> Path:
    """Write prediction scores to the Codabench CSV format.

    Args:
        predictions (dict[str, float]): Mapping from capture ID to prediction score.
        output_path (str | Path): Destination CSV path. Defaults to "predictions.csv".

    Returns:
        Path: Path to the generated CSV file.

    Raises:
        ValueError: If capture IDs are empty or scores fall outside [0, 1].
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, float]] = []
    for capture_id, score in predictions.items():
        if capture_id is None or not str(capture_id).strip():
            raise ValueError("Invalid capture_id in predictions: empty value")
        score_value = float(score)
        if not (0.0 <= score_value <= 1.0):
            raise ValueError(
                f"Invalid prediction score for capture_id={capture_id}: {score_value}. Expected [0, 1]."
            )
        rows.append((str(capture_id), score_value))

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["capture_id", "pred"])
        writer.writerows(rows)

    return output


def validate_csv(csv_path: str | Path) -> bool:
    """Validate a prediction CSV against expected schema and value ranges.

    Args:
        csv_path (str | Path): Path to the prediction CSV file.

    Returns:
        bool: True when CSV content is valid.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If header, row width, or prediction values are invalid.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {path}")

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(
                "CSV is empty. Expected header and prediction rows."
            ) from exc

        if len(header) != 2 or header[0] != "capture_id" or header[1] != "pred":
            raise ValueError("Invalid CSV header. Expected exactly: capture_id,pred")

        count = 0
        for i, row in enumerate(reader, start=2):
            if len(row) != 2:
                raise ValueError(
                    f"Invalid row format at line {i}. Expected 2 columns, got {len(row)}."
                )
            capture_id, pred_text = row
            if not capture_id.strip():
                raise ValueError(f"Empty capture_id at line {i}.")
            try:
                pred_value = float(pred_text)
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric pred value at line {i}: {pred_text!r}"
                ) from exc
            if not (0.0 <= pred_value <= 1.0):
                raise ValueError(
                    f"Pred out of range at line {i}: {pred_value}. Expected [0, 1]."
                )
            count += 1

    print(f"CSV validation successful. Number of predictions: {count}")
    return True


def prepare_submission(
    csv_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Package a validated prediction CSV into a zip archive.

    Args:
        csv_path (str | Path): Path to the prediction CSV file.
        output_path (str | Path | None): Optional destination zip path.

    Returns:
        Path: Path to the generated zip archive.
    """
    csv_file = Path(csv_path)
    validate_csv(csv_file)

    if output_path is None:
        zip_path = csv_file.with_suffix(".zip")
    else:
        zip_path = Path(output_path)
        if zip_path.suffix != ".zip":
            zip_path = zip_path.with_suffix(".zip")

    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_file, arcname=csv_file.name)

    return zip_path
