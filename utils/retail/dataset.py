# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Dataset utilities for Kaputt retail track data loading.

Aligned with the official Kaputt reference loader: parquet files live at
``root/query-{split}.parquet`` and ``root/reference-{split}.parquet``, and
image paths stored in parquet columns are resolved relative to ``root``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from retail.types import (
    RetailBatch,
    RetailInferenceBatch,
    RetailInferenceSample,
    RetailSample,
    Split,
)


def _default_transform(image_size: tuple[int, int] = (224, 224)) -> Any:
    """Return the default image preprocessing transform."""
    return transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])


def _pad_tensor(tensor: torch.Tensor, target_size: int, pad_value: float = 0) -> torch.Tensor:
    """Pad or truncate the first dimension of *tensor* to *target_size*.

    Matches the official Kaputt reference ``pad_tensor`` implementation.
    """
    pad_size = target_size - tensor.size(0)
    if pad_size <= 0:
        return tensor[:target_size]
    return torch.cat([tensor, torch.full((pad_size, *tensor.size()[1:]), pad_value)])


class KaputtDataset(Dataset):
    """Kaputt training or validation dataset.

    Follows the official Kaputt reference loader: reads query and reference
    parquet files from ``root/query-{split}.parquet`` and
    ``root/reference-{split}.parquet``, resolves image paths directly from
    parquet column values, and returns query + reference images, crops, and
    masks in each sample.

    Args:
        root: Root directory containing Kaputt track data.
        split: Dataset split. Defaults to ``Split.TRAIN``.
        transform: Transform applied to each loaded image. Defaults to
            resize-to-224 and tensor conversion.
        image_size: Target ``(H, W)`` when using the default transform.
            Ignored if *transform* is provided. Defaults to ``(224, 224)``.
        max_references: Maximum number of reference images per sample.
            Shorter sequences are zero-padded; longer ones are truncated.
            Defaults to 3.
    """

    def __init__(
        self,
        root: str | Path,
        split: Split | str = Split.TRAIN,
        transform: Any = None,
        image_size: tuple[int, int] = (224, 224),
        max_references: int = 3,
    ) -> None:
        self.root = Path(root)
        self.split = Split(split)
        self.max_references = max_references
        self.transform = transform or _default_transform(image_size)

        query_parquet = self.root / f"query-{self.split}.parquet"
        if not query_parquet.exists():
            raise FileNotFoundError(f"Query parquet not found: {query_parquet}")
        self.query_data = pd.read_parquet(query_parquet)

        reference_parquet = self.root / f"reference-{self.split}.parquet"
        self.reference_data = (
            pd.read_parquet(reference_parquet)
            if reference_parquet.exists()
            else pd.DataFrame()
        )

    def __len__(self) -> int:
        return len(self.query_data)

    def __getitem__(self, index: int) -> RetailSample:
        query_row = self.query_data.iloc[index]

        # Load query images and mask
        query_image = Image.open(self.root / query_row.query_image)
        query_crop = Image.open(self.root / query_row.query_crop)
        query_mask = Image.open(self.root / query_row.query_mask)

        # Load reference images for this item
        ref_rows = self.reference_data[
            self.reference_data.item_identifier == query_row.item_identifier
        ]
        ref_images = [Image.open(self.root / row.reference_image) for _, row in ref_rows.iterrows()]
        ref_crops = [Image.open(self.root / row.reference_crop) for _, row in ref_rows.iterrows()]
        ref_masks = [Image.open(self.root / row.reference_mask) for _, row in ref_rows.iterrows()]

        # Apply transforms
        query_image_t = self.transform(query_image)
        query_crop_t = self.transform(query_crop)
        query_mask_t = self.transform(query_mask)
        ref_image_t = torch.stack([self.transform(img) for img in ref_images]) if ref_images else torch.empty(0)
        ref_crop_t = torch.stack([self.transform(c) for c in ref_crops]) if ref_crops else torch.empty(0)
        ref_mask_t = torch.stack([self.transform(m) for m in ref_masks]) if ref_masks else torch.empty(0)

        # Pad reference tensors to max_references
        ref_image_t = _pad_tensor(ref_image_t, self.max_references)
        ref_crop_t = _pad_tensor(ref_crop_t, self.max_references)
        ref_mask_t = _pad_tensor(ref_mask_t, self.max_references)

        return RetailSample(
            query_image=query_image_t,
            query_crop=query_crop_t,
            query_mask=query_mask_t,
            reference_image=ref_image_t,
            reference_crop=ref_crop_t,
            reference_mask=ref_mask_t,
            item_material=query_row.item_material,
            defect=bool(query_row.defect),
            major_defect=bool(query_row.major_defect),
            defect_types=query_row.defect_types,
            capture_id=query_row.capture_id,
            item_identifier=query_row.item_identifier,
        )


class Kaputt2Dataset(Dataset):
    """Kaputt 2 inference dataset.

    Follows the same path conventions as :class:`KaputtDataset` but does not
    require ground-truth labels (``defect``, ``major_defect``, etc.).

    Args:
        root: Root directory containing Kaputt 2 track data.
        split: Dataset split. Defaults to ``Split.TEST``.
        transform: Transform applied to each loaded image.
        image_size: Target ``(H, W)`` for the default transform.
        max_references: Maximum reference images per sample.
    """

    def __init__(
        self,
        root: str | Path,
        split: Split | str = Split.TEST,
        transform: Any = None,
        image_size: tuple[int, int] = (224, 224),
        max_references: int = 3,
    ) -> None:
        self.root = Path(root)
        self.split = Split(split)
        self.max_references = max_references
        self.transform = transform or _default_transform(image_size)

        query_parquet = self.root / f"query-{self.split}.parquet"
        if not query_parquet.exists():
            raise FileNotFoundError(f"Query parquet not found: {query_parquet}")
        self.query_data = pd.read_parquet(query_parquet)

        reference_parquet = self.root / f"reference-{self.split}.parquet"
        self.reference_data = (
            pd.read_parquet(reference_parquet)
            if reference_parquet.exists()
            else pd.DataFrame()
        )

    def __len__(self) -> int:
        return len(self.query_data)

    def __getitem__(self, index: int) -> RetailInferenceSample:
        query_row = self.query_data.iloc[index]

        query_image = Image.open(self.root / query_row.query_image)
        query_crop = Image.open(self.root / query_row.query_crop)
        query_mask = Image.open(self.root / query_row.query_mask)

        ref_rows = self.reference_data[
            self.reference_data.item_identifier == query_row.item_identifier
        ]
        ref_images = [Image.open(self.root / row.reference_image) for _, row in ref_rows.iterrows()]
        ref_crops = [Image.open(self.root / row.reference_crop) for _, row in ref_rows.iterrows()]
        ref_masks = [Image.open(self.root / row.reference_mask) for _, row in ref_rows.iterrows()]

        query_image_t = self.transform(query_image)
        query_crop_t = self.transform(query_crop)
        query_mask_t = self.transform(query_mask)
        ref_image_t = torch.stack([self.transform(img) for img in ref_images]) if ref_images else torch.empty(0)
        ref_crop_t = torch.stack([self.transform(c) for c in ref_crops]) if ref_crops else torch.empty(0)
        ref_mask_t = torch.stack([self.transform(m) for m in ref_masks]) if ref_masks else torch.empty(0)

        ref_image_t = _pad_tensor(ref_image_t, self.max_references)
        ref_crop_t = _pad_tensor(ref_crop_t, self.max_references)
        ref_mask_t = _pad_tensor(ref_mask_t, self.max_references)

        return RetailInferenceSample(
            query_image=query_image_t,
            query_crop=query_crop_t,
            query_mask=query_mask_t,
            reference_image=ref_image_t,
            reference_crop=ref_crop_t,
            reference_mask=ref_mask_t,
            capture_id=query_row.capture_id,
            item_identifier=query_row.item_identifier,
        )


def _collate_retail_batch(items: list[RetailSample]) -> RetailBatch:
    """Collate retail training/validation samples into a mini-batch."""
    return RetailBatch(
        query_image=torch.stack([s.query_image for s in items]),
        query_crop=torch.stack([s.query_crop for s in items]),
        query_mask=torch.stack([s.query_mask for s in items]),
        reference_image=torch.stack([s.reference_image for s in items]),
        reference_crop=torch.stack([s.reference_crop for s in items]),
        reference_mask=torch.stack([s.reference_mask for s in items]),
        item_material=[s.item_material for s in items],
        defect=torch.tensor([s.defect for s in items], dtype=torch.bool),
        major_defect=torch.tensor([s.major_defect for s in items], dtype=torch.bool),
        defect_types=[s.defect_types for s in items],
        capture_id=[s.capture_id for s in items],
        item_identifier=[s.item_identifier for s in items],
    )


def _collate_inference_batch(items: list[RetailInferenceSample]) -> RetailInferenceBatch:
    """Collate retail inference samples into a mini-batch."""
    return RetailInferenceBatch(
        query_image=torch.stack([s.query_image for s in items]),
        query_crop=torch.stack([s.query_crop for s in items]),
        query_mask=torch.stack([s.query_mask for s in items]),
        reference_image=torch.stack([s.reference_image for s in items]),
        reference_crop=torch.stack([s.reference_crop for s in items]),
        reference_mask=torch.stack([s.reference_mask for s in items]),
        capture_id=[s.capture_id for s in items],
        item_identifier=[s.item_identifier for s in items],
    )


def get_dataloader(
    root: str | Path,
    split: Split | str,
    batch_size: int = 32,
    num_workers: int = 4,
    transform: Any = None,
    image_size: tuple[int, int] = (224, 224),
    max_references: int = 3,
) -> DataLoader:
    """Build a Kaputt training/validation dataloader.

    Args:
        root: Root directory containing Kaputt track data.
        split: Dataset split.
        batch_size: Batch size. Defaults to 32.
        num_workers: Worker process count. Defaults to 4.
        transform: Optional image transform.
        image_size: Target image size for the default transform.
        max_references: Max reference images per sample.

    Returns:
        Configured dataloader yielding :class:`RetailBatch`.
    """
    split = Split(split)
    dataset = KaputtDataset(
        root=root,
        split=split,
        transform=transform,
        image_size=image_size,
        max_references=max_references,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == Split.TRAIN,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_retail_batch,
    )


def get_inference_dataloader(
    root: str | Path,
    split: Split | str = Split.TEST,
    batch_size: int = 32,
    num_workers: int = 4,
    transform: Any = None,
    image_size: tuple[int, int] = (224, 224),
    max_references: int = 3,
) -> DataLoader:
    """Build a Kaputt 2 inference dataloader.

    Args:
        root: Root directory containing Kaputt 2 track data.
        split: Dataset split. Defaults to ``Split.TEST``.
        batch_size: Batch size. Defaults to 32.
        num_workers: Worker process count. Defaults to 4.
        transform: Optional image transform.
        image_size: Target image size for the default transform.
        max_references: Max reference images per sample.

    Returns:
        Configured dataloader yielding :class:`RetailInferenceBatch`.
    """
    dataset = Kaputt2Dataset(
        root=root,
        split=split,
        transform=transform,
        image_size=image_size,
        max_references=max_references,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_inference_batch,
    )
