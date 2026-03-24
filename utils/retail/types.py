# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Enums and data containers for the retail track."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch


class Split(StrEnum):
    """Kaputt dataset split identifiers."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass
class RetailSample:
    """Single query sample from the Kaputt training/validation dataset.

    Mirrors the official Kaputt reference loader output: query images,
    reference images (padded to ``max_references``), masks, and metadata.
    """

    query_image: torch.Tensor
    query_crop: torch.Tensor
    query_mask: torch.Tensor
    reference_image: torch.Tensor  # (max_references, C, H, W)
    reference_crop: torch.Tensor  # (max_references, C, H, W)
    reference_mask: torch.Tensor  # (max_references, C, H, W)
    item_material: str
    defect: bool
    major_defect: bool
    defect_types: str
    capture_id: str
    item_identifier: str


@dataclass
class RetailInferenceSample:
    """Single inference sample from the Kaputt 2 test dataset."""

    query_image: torch.Tensor
    query_crop: torch.Tensor
    query_mask: torch.Tensor
    reference_image: torch.Tensor  # (max_references, C, H, W)
    reference_crop: torch.Tensor  # (max_references, C, H, W)
    reference_mask: torch.Tensor  # (max_references, C, H, W)
    capture_id: str
    item_identifier: str


@dataclass
class RetailBatch:
    """Collated mini-batch of retail training/validation samples."""

    query_image: torch.Tensor
    query_crop: torch.Tensor
    query_mask: torch.Tensor
    reference_image: torch.Tensor  # (B, max_references, C, H, W)
    reference_crop: torch.Tensor  # (B, max_references, C, H, W)
    reference_mask: torch.Tensor  # (B, max_references, C, H, W)
    item_material: list[str]
    defect: torch.Tensor
    major_defect: torch.Tensor
    defect_types: list[str]
    capture_id: list[str]
    item_identifier: list[str]


@dataclass
class RetailInferenceBatch:
    """Collated mini-batch of retail inference samples."""

    query_image: torch.Tensor
    query_crop: torch.Tensor
    query_mask: torch.Tensor
    reference_image: torch.Tensor  # (B, max_references, C, H, W)
    reference_crop: torch.Tensor  # (B, max_references, C, H, W)
    reference_mask: torch.Tensor  # (B, max_references, C, H, W)
    capture_id: list[str]
    item_identifier: list[str]


@dataclass
class RetailEvalResult:
    """Result of local retail track evaluation."""

    ap_any: float
    ap_major: float
    auroc: float
    recall_at_50p: float
    recall_at_1fpr: float
