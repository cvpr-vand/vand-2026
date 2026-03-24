# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Local evaluation utilities for Kaputt retail predictions.

Note:
    The metric utilities in this module are provided for convenience and local
    testing only. Official scores are computed by the respective challenge
    servers, whose implementations may differ. No claims or entitlements can be
    derived from the local evaluation results.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from retail.types import RetailEvalResult


def _load_predictions(predictions: dict[str, float] | str | Path) -> dict[str, float]:
    """Load predictions from mapping or CSV file."""
    if isinstance(predictions, dict):
        return {str(k): float(v) for k, v in predictions.items()}

    csv_path = Path(predictions)
    if not csv_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    expected_columns = ["capture_id", "pred"]
    if list(df.columns) != expected_columns:
        raise ValueError(
            f"Invalid CSV columns. Expected {expected_columns}, got {list(df.columns)}"
        )

    out: dict[str, float] = {}
    for row in df.itertuples(index=False):
        out[str(row.capture_id)] = float(row.pred)
    return out


def _safe_auroc(y_true, y_score) -> float:
    """Compute AUROC while handling single-class targets."""
    if y_true.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _recall_at_precision(y_true, y_score, precision_target: float) -> float:
    """Compute maximum recall under a minimum precision constraint."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    eligible = recall[precision >= precision_target]
    if eligible.size == 0:
        return 0.0
    return float(eligible.max())


def _recall_at_fpr(y_true, y_score, fpr_target: float) -> float:
    """Compute maximum recall under a maximum false-positive-rate constraint."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    eligible = tpr[fpr <= fpr_target]
    if eligible.size == 0:
        return 0.0
    return float(eligible.max())


def evaluate_local(
    predictions: dict[str, float] | str | Path,
    dataset_root: str | Path,
) -> RetailEvalResult:
    """Evaluate local predictions against Kaputt test ground truth.

    Args:
        predictions (dict[str, float] | str | Path): Prediction mapping or CSV path.
        dataset_root (str | Path): Root path containing datasets/query-test.parquet.

    Returns:
        RetailEvalResult: Evaluation scores containing AP, AUROC, and recall metrics.

    Raises:
        FileNotFoundError: If prediction CSV or ground-truth parquet is missing.
        ValueError: If input CSV schema is invalid or predictions are incomplete.
    """
    preds = _load_predictions(predictions)

    root = Path(dataset_root)
    parquet_path = root / "query-test.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Ground truth parquet not found: {parquet_path}")

    gt = pd.read_parquet(parquet_path)
    needed = {"capture_id", "defect", "major_defect"}
    missing = needed.difference(gt.columns)
    if missing:
        raise ValueError(f"Missing required ground-truth columns: {sorted(missing)}")

    gt = gt[["capture_id", "defect", "major_defect"]].copy()
    gt["capture_id"] = gt["capture_id"].astype(str)
    gt["pred"] = gt["capture_id"].map(preds)

    missing_preds = gt["pred"].isna().sum()
    if missing_preds > 0:
        raise ValueError(f"Missing predictions for {missing_preds} capture_ids")

    y_any = gt["defect"].astype(bool).astype(int)
    y_score = gt["pred"].astype(float)

    ap_any = float(average_precision_score(y_any, y_score))
    auroc_any = _safe_auroc(y_any, y_score)
    r_at_50p = _recall_at_precision(y_any, y_score, precision_target=0.5)
    r_at_1fpr = _recall_at_fpr(y_any, y_score, fpr_target=0.01)

    major = gt["major_defect"].astype(bool)
    defect = gt["defect"].astype(bool)
    major_mask = major | (~defect)

    y_major = major[major_mask].astype(int)
    s_major = y_score[major_mask]
    ap_major = float(average_precision_score(y_major, s_major))

    result = RetailEvalResult(
        ap_any=ap_any,
        ap_major=ap_major,
        auroc=auroc_any,
        recall_at_50p=r_at_50p,
        recall_at_1fpr=r_at_1fpr,
    )

    print("\nLocal Evaluation (Kaputt1 test)")
    print("-" * 38)
    for field_name, value in [
        ("AP_any", result.ap_any),
        ("AP_major", result.ap_major),
        ("AUROC", result.auroc),
        ("R@50%P", result.recall_at_50p),
        ("R@1%FPR", result.recall_at_1fpr),
    ]:
        print(f"{field_name:<10} {value:>10.6f}")
    print("-" * 38)

    return result
