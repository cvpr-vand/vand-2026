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
    """Single query sample from the Kaputt training/validation dataset."""

    image: torch.Tensor
    capture_id: str
    item_identifier: str
    label: int
    is_major: bool
    defect_types: list[str]
    image_path: str


@dataclass
class RetailInferenceSample:
    """Single inference sample from the Kaputt 2 test dataset."""

    image: torch.Tensor
    capture_id: str
    item_identifier: str
    image_path: str


@dataclass
class RetailBatch:
    """Collated mini-batch of retail training/validation samples."""

    image: torch.Tensor
    capture_id: list[str]
    item_identifier: list[str]
    label: torch.Tensor
    is_major: list[bool]
    defect_types: list[list[str]]
    image_path: list[str]


@dataclass
class RetailInferenceBatch:
    """Collated mini-batch of retail inference samples."""

    image: torch.Tensor
    capture_id: list[str]
    item_identifier: list[str]
    image_path: list[str]


@dataclass
class RetailEvalResult:
    """Result of local retail track evaluation."""

    ap_any: float
    ap_major: float
    auroc: float
    recall_at_50p: float
    recall_at_1fpr: float
