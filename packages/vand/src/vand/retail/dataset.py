# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Dataset utilities for Kaputt retail track data loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from vand.retail.types import (
    RetailBatch,
    RetailInferenceBatch,
    RetailInferenceSample,
    RetailSample,
    Split,
)


def _default_transform() -> Any:
    """Return the default image preprocessing transform."""
    return transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])


def _parse_defect_types(value: Any) -> list[str]:
    """Parse defect types from parquet metadata into a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []

    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()

    separator = "," if "," in text else ";"
    parts = [p.strip().strip("\"'") for p in text.split(separator)]
    return [p for p in parts if p]


def _open_rgb(path: Path) -> Image.Image:
    """Open an image file as RGB."""
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    return Image.open(path).convert("RGB")


class KaputtDataset(Dataset):
    """Kaputt training or validation dataset.

    Args:
        root (str | Path): Root directory containing Kaputt track data.
        split (Split | str): Dataset split. Defaults to ``Split.TRAIN``.
        use_crops (bool): Whether to use crop images instead of full images.
            Defaults to False.
        transform (Any): Transform applied to each loaded image. Defaults to
            torchvision resize-to-256 and tensor conversion when None.
    """

    def __init__(
        self,
        root: str | Path,
        split: Split | str = Split.TRAIN,
        use_crops: bool = False,
        transform: Any = None,
    ) -> None:
        self.root = Path(root)
        self.split = Split(split)
        self.use_crops = use_crops
        self.transform = transform or _default_transform()

        parquet_path = self.root / "datasets" / f"query-{split}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet metadata not found: {parquet_path}")

        self.df = pd.read_parquet(parquet_path).reset_index(drop=True)

        reference_parquet = self.root / "datasets" / f"reference-{split}.parquet"
        self.reference_df = (
            pd.read_parquet(reference_parquet).reset_index(drop=True) if reference_parquet.exists() else pd.DataFrame()
        )

    def __len__(self) -> int:
        """Return the number of query samples.

        Returns:
            int: Number of rows in query parquet metadata.
        """
        return len(self.df)

    def _query_image_path(self, row: Any) -> Path:
        """Resolve the query image path for a row with fallback."""
        image_type = "crop" if self.use_crops else "image"
        parquet_key = f"query_{image_type}"
        prefix = self.root / f"query-{image_type}"

        parquet_value = row.get(parquet_key)
        if parquet_value is not None and str(parquet_value).strip():
            candidate = prefix / str(parquet_value)
            if candidate.exists():
                return candidate

        capture_id = str(row["capture_id"])
        fallback = prefix / "data" / self.split / "query-data" / image_type / f"{capture_id}.jpg"
        return fallback

    def _reference_image_path(self, row: Any) -> Path:
        """Resolve a reference image path for a row with fallback."""
        image_type = "crop" if self.use_crops else "image"
        parquet_key = f"reference_{image_type}"
        prefix = self.root / f"reference-{image_type}"

        parquet_value = row.get(parquet_key)
        if parquet_value is not None and str(parquet_value).strip():
            candidate = prefix / str(parquet_value)
            if candidate.exists():
                return candidate

        capture_id = str(row["capture_id"])
        fallback = prefix / "data" / self.split / "reference-data" / image_type / f"{capture_id}.jpg"
        return fallback

    def __getitem__(self, index: int) -> RetailSample:
        """Fetch a query sample.

        Args:
            index (int): Sample index in the dataset.

        Returns:
            RetailSample: Sample with image tensor, metadata, and labels.

        Raises:
            FileNotFoundError: If resolved image path does not exist.
        """
        row = self.df.iloc[index]
        image_path = self._query_image_path(row)
        image = _open_rgb(image_path)
        image_tensor = self.transform(image)

        defect = bool(row.get("defect", False))
        major = bool(row.get("major_defect", False))

        return RetailSample(
            image=image_tensor,
            capture_id=str(row["capture_id"]),
            item_identifier=str(row["item_identifier"]),
            label=int(defect),
            is_major=major,
            defect_types=_parse_defect_types(row.get("defect_types")),
            image_path=str(image_path),
        )

    def get_reference_images(self, item_identifier: str) -> list[Any]:
        """Return transformed reference images for an item identifier.

        Args:
            item_identifier (str): Item identifier used to filter rows.

        Returns:
            list[Any]: List of transformed reference images.
        """
        rows = self.reference_df[self.reference_df["item_identifier"].astype(str) == str(item_identifier)]
        if rows.empty:
            rows = self.df[self.df["item_identifier"].astype(str) == str(item_identifier)]
        out: list[Any] = []
        seen: set[str] = set()
        for _, row in rows.iterrows():
            capture_id = str(row["capture_id"])
            if capture_id in seen:
                continue
            seen.add(capture_id)
            path = self._reference_image_path(row)
            if path.exists():
                out.append(self.transform(_open_rgb(path)))
        return out


class Kaputt2Dataset(Dataset):
    """Kaputt inference dataset for test split.

    Args:
        root (str | Path): Root directory containing Kaputt track data.
        split (Split | str): Dataset split. Defaults to ``Split.TEST``.
        use_crops (bool): Whether to use crop images instead of full images.
            Defaults to False.
        transform (Any): Transform applied to each loaded image. Defaults to
            torchvision resize-to-256 and tensor conversion when None.
    """

    def __init__(
        self,
        root: str | Path,
        split: Split | str = Split.TEST,
        use_crops: bool = False,
        transform: Any = None,
    ) -> None:
        self.root = Path(root)
        self.split = Split(split)
        self.use_crops = use_crops
        self.transform = transform or _default_transform()

        parquet_path = self.root / "datasets" / f"query-{split}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet metadata not found: {parquet_path}")
        self.df = pd.read_parquet(parquet_path).reset_index(drop=True)

    def __len__(self) -> int:
        """Return the number of query samples.

        Returns:
            int: Number of rows in query parquet metadata.
        """
        return len(self.df)

    def _query_image_path(self, row: Any) -> Path:
        """Resolve the query image path for a row with fallback."""
        image_type = "crop" if self.use_crops else "image"
        parquet_key = f"query_{image_type}"
        prefix = self.root / f"query-{image_type}"

        parquet_value = row.get(parquet_key)
        if parquet_value is not None and str(parquet_value).strip():
            candidate = prefix / str(parquet_value)
            if candidate.exists():
                return candidate

        capture_id = str(row["capture_id"])
        fallback = prefix / "data" / self.split / "query-data" / image_type / f"{capture_id}.jpg"
        return fallback

    def __getitem__(self, index: int) -> RetailInferenceSample:
        """Fetch an inference sample.

        Args:
            index (int): Sample index in the dataset.

        Returns:
            RetailInferenceSample: Sample with image tensor and metadata.

        Raises:
            FileNotFoundError: If resolved image path does not exist.
        """
        row = self.df.iloc[index]
        image_path = self._query_image_path(row)
        image = _open_rgb(image_path)
        image_tensor = self.transform(image)

        return RetailInferenceSample(
            image=image_tensor,
            capture_id=str(row["capture_id"]),
            item_identifier=str(row["item_identifier"]),
            image_path=str(image_path),
        )


def _collate_retail_batch(items: list[RetailSample]) -> RetailBatch:
    """Collate retail training/validation samples into a mini-batch."""
    return RetailBatch(
        image=torch.stack([item.image for item in items], dim=0),
        capture_id=[item.capture_id for item in items],
        item_identifier=[item.item_identifier for item in items],
        label=torch.tensor([item.label for item in items], dtype=torch.int64),
        is_major=[item.is_major for item in items],
        defect_types=[item.defect_types for item in items],
        image_path=[item.image_path for item in items],
    )


def _collate_inference_batch(items: list[RetailInferenceSample]) -> RetailInferenceBatch:
    """Collate retail inference samples into a mini-batch."""
    return RetailInferenceBatch(
        image=torch.stack([item.image for item in items], dim=0),
        capture_id=[item.capture_id for item in items],
        item_identifier=[item.item_identifier for item in items],
        image_path=[item.image_path for item in items],
    )


def get_dataloader(
    root: str | Path,
    split: Split | str,
    batch_size: int = 32,
    num_workers: int = 4,
    use_crops: bool = False,
    transform: Any = None,
) -> DataLoader:
    """Build a Kaputt training/validation dataloader.

    Args:
        root (str | Path): Root directory containing Kaputt track data.
        split (Split | str): Dataset split.
        batch_size (int): Batch size for loading samples. Defaults to 32.
        num_workers (int): Number of data loader workers. Defaults to 4.
        use_crops (bool): Whether to use crop images. Defaults to False.
        transform (Any): Optional image transform. Defaults to None.

    Returns:
        DataLoader: Configured dataloader yielding :class:`RetailBatch`.
    """
    split = Split(split)
    dataset = KaputtDataset(
        root=root,
        split=split,
        use_crops=use_crops,
        transform=transform,
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
    use_crops: bool = False,
    transform: Any = None,
) -> DataLoader:
    """Build a Kaputt 2 inference dataloader.

    Args:
        root (str | Path): Root directory containing Kaputt 2 track data.
        split (Split | str): Dataset split. Defaults to ``Split.TEST``.
        batch_size (int): Batch size for loading samples. Defaults to 32.
        num_workers (int): Number of data loader workers. Defaults to 4.
        use_crops (bool): Whether to use crop images. Defaults to False.
        transform (Any): Optional image transform. Defaults to None.

    Returns:
        DataLoader: Configured dataloader yielding :class:`RetailInferenceBatch`.
    """
    dataset = Kaputt2Dataset(
        root=root,
        split=split,
        use_crops=use_crops,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_inference_batch,
    )
