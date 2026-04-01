# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Submission file generation and validation utilities."""

import csv
import io
import zipfile
from pathlib import Path

import pandas as pd


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

    with path.open("r", newline="", encoding="utf-8-sig") as f:
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


def _load_ground_truth_ids(path: Path) -> set[str]:
    """Return the set of capture_ids from a ground truth parquet file.

    Args:
        path (Path): Path to the ground truth parquet file.

    Returns:
        set[str]: Unique capture IDs found in the ground truth.

    Raises:
        FileNotFoundError: If the parquet file does not exist.
        ValueError: If the parquet file is missing the 'capture_id' column.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Ground truth file not found: {path}")
    df = pd.read_parquet(path, columns=["capture_id"])
    if "capture_id" not in df.columns:
        raise ValueError(f"Ground truth file missing 'capture_id' column: {path}")
    return set(df["capture_id"].astype(str).unique())


def validate_submission_zip(
    zip_path: str | Path,
    ground_truth_path: str | Path,
) -> bool:
    """Validate a Kaputt 2 submission zip file before uploading.

    The zip must contain exactly one CSV at its root with columns
    ``capture_id`` and ``pred``. Predictions are checked for numeric type
    and [0, 1] range. Coverage against the ground truth parquet is verified.

    Args:
        zip_path (str | Path): Path to the submission zip file.
        ground_truth_path (str | Path): Path to the ground truth parquet file
            containing expected capture_ids.

    Returns:
        bool: True when the submission passes all checks.

    Raises:
        FileNotFoundError: If the zip file or ground truth file does not exist.
        ValueError: If the zip structure, CSV schema, values, or coverage
            are invalid.
    """
    zip_file = Path(zip_path)
    gt_file = Path(ground_truth_path)

    # 1. Check zip exists and is valid
    if not zip_file.is_file():
        raise FileNotFoundError(f"Submission zip not found: {zip_file}")
    if not zipfile.is_zipfile(zip_file):
        raise ValueError(f"Not a valid zip file: {zip_file}")

    # 2. Extract CSV from zip
    with zipfile.ZipFile(zip_file, "r") as zf:
        csv_files = [
            n
            for n in zf.namelist()
            if n.lower().endswith(".csv") and not n.startswith("__MACOSX")
        ]
        if len(csv_files) == 0:
            raise ValueError("Zip contains no CSV files.")
        if len(csv_files) > 1:
            raise ValueError(
                f"Zip contains {len(csv_files)} CSV files — expected exactly 1."
            )
        csv_name = csv_files[0]
        if "/" in csv_name or "\\" in csv_name:
            raise ValueError(
                f"CSV file '{csv_name}' is inside a subfolder — it must be at the root of the zip."
            )
        raw = zf.read(csv_name)

    # 3. Decode UTF-8
    try:
        csv_text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("CSV is not valid UTF-8.") from exc

    # 4. Parse CSV
    try:
        reader = csv.DictReader(io.StringIO(csv_text))
        columns = reader.fieldnames or []
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV: {exc}") from exc

    # 5. Check columns
    required = {"capture_id", "pred"}
    col_set = set(columns)
    missing_cols = required - col_set
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")
    extra = col_set - required
    if extra:
        raise ValueError(
            f"Unexpected columns: {sorted(extra)}. Only 'capture_id' and 'pred' are allowed."
        )
    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate column names detected.")

    # 6. Read rows and validate values
    rows = list(reader)
    if len(rows) == 0:
        raise ValueError("CSV has no data rows.")

    gt_ids = _load_ground_truth_ids(gt_file)

    capture_ids: list[str] = []
    bad_pred_lines: list[int] = []
    out_of_range_lines: list[int] = []

    for i, row in enumerate(rows, start=2):  # line 2 = first data row
        # DictReader sets missing fields to None and uses None as key for
        # extra fields.  Reject malformed rows before accessing values.
        if None in row or None in row.values():
            raise ValueError(
                f"Row {i} has the wrong number of fields (expected {len(columns)})."
            )

        cid = (row.get("capture_id") or "").strip()
        if not cid:
            raise ValueError(f"Empty capture_id at row {i}.")
        capture_ids.append(cid)

        pred = (row.get("pred") or "").strip()
        if not pred:
            bad_pred_lines.append(i)
            continue
        try:
            val = float(pred)
            if not (0.0 <= val <= 1.0):
                out_of_range_lines.append(i)
        except ValueError:
            bad_pred_lines.append(i)

    if bad_pred_lines:
        sample = bad_pred_lines[:5]
        suffix = (
            f" (and {len(bad_pred_lines) - 5} more)" if len(bad_pred_lines) > 5 else ""
        )
        raise ValueError(
            f"Non-numeric or empty 'pred' values on lines: {sample}{suffix}"
        )

    if out_of_range_lines:
        sample = out_of_range_lines[:5]
        suffix = (
            f" (and {len(out_of_range_lines) - 5} more)"
            if len(out_of_range_lines) > 5
            else ""
        )
        raise ValueError(
            f"'pred' values outside [0, 1] range on lines: {sample}{suffix}"
        )

    # 7. Duplicate capture_ids
    seen: set[str] = set()
    dupes: set[str] = set()
    for cid in capture_ids:
        if cid in seen:
            dupes.add(cid)
        seen.add(cid)
    if dupes:
        sample = sorted(dupes)[:3]
        suffix = f" (and {len(dupes) - 3} more)" if len(dupes) > 3 else ""
        raise ValueError(f"Duplicate capture_ids found: {sample}{suffix}")

    # 8. Check row count against ground truth (after dedup)
    unique_ids = set(capture_ids)
    if len(unique_ids) > len(gt_ids):
        raise ValueError(
            f"CSV has {len(unique_ids)} unique capture_ids. Max allowed is "
            f"{len(gt_ids)} (unique capture_ids in ground truth)."
        )

    # 9. Check coverage against ground truth
    unknown = unique_ids - gt_ids
    missing = gt_ids - unique_ids

    if unknown:
        sample = sorted(unknown)[:3]
        raise ValueError(f"{len(unknown)} capture_ids not in ground truth: {sample}...")

    if missing:
        matched = unique_ids & gt_ids
        raise ValueError(
            f"{len(missing)} ground truth capture_ids missing from submission "
            f"({len(matched)}/{len(gt_ids)} coverage). "
            f"All {len(gt_ids)} capture_ids are required."
        )

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
