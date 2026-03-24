"""Model skeleton for industrial track participants.

Implement your anomaly detection model by filling in the methods below.
See examples/industrial/ for a working baseline implementation.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class AnomalyModel(nn.Module):
    """Anomaly detection model for the industrial track.

    Participants must implement all methods marked with ``NotImplementedError``.
    The train, test, and submit entrypoints call these methods, so the interface
    must be preserved.

    Args:
        device (str | None): Target device. Defaults to CUDA when available.
    """

    def __init__(self, device: str | None = None) -> None:
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def fit(self, train_dataloader: Any) -> None:
        """Train the model on normal images.

        Args:
            train_dataloader (Any): Dataloader yielding ``IndustrialBatch`` instances
                with fields ``image`` (tensor) and ``label`` (int, always 0 for train split).
        """
        raise NotImplementedError

    def predict(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict pixel-level anomaly map and image-level score for one image.

        Args:
            image (torch.Tensor): Single image tensor of shape ``(C, H, W)`` or ``(1, C, H, W)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Anomaly map ``(H, W)`` and scalar anomaly score,
                both on CPU.
        """
        raise NotImplementedError

    def get_threshold(self, val_dataloader: Any) -> float:
        """Compute binarization threshold from validation data.

        Args:
            val_dataloader (Any): Dataloader yielding validation batches.

        Returns:
            float: Threshold value for binarizing anomaly maps.
        """
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        """Save model state to disk.

        Args:
            path (str | Path): Output checkpoint path.
        """
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> AnomalyModel:
        """Load model state from disk.

        Args:
            path (str | Path): Checkpoint path.
            device (str | None): Optional device override.

        Returns:
            AnomalyModel: Loaded model instance.
        """
        raise NotImplementedError
