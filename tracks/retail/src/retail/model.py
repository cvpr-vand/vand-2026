"""Model skeleton for retail track participants.

Implement your defect detection model by filling in the methods below.
See examples/retail/ for a working baseline implementation.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class DefectModel(nn.Module):
    """Defect detection model for the retail track.

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

    def fit(self, train_dataloader: Any, reference_fn: Any = None) -> DefectModel:
        """Train the model on Kaputt data.

        Args:
            train_dataloader (Any): Dataloader yielding ``RetailBatch`` instances with
                fields ``image``, ``label``, ``capture_id``, ``item_identifier``.
            reference_fn (Any): Optional callable that takes an item_identifier string
                and returns a list of reference image tensors for that item.

        Returns:
            DefectModel: Fitted model instance.
        """
        raise NotImplementedError

    def predict(self, image: torch.Tensor, reference_embeddings: list[Any] | None = None) -> float:
        """Predict a defect score for a single image.

        Args:
            image (torch.Tensor): Single image tensor of shape ``(C, H, W)``.
            reference_embeddings (list[Any] | None): Optional precomputed reference embeddings.

        Returns:
            float: Defect score in [0, 1], where higher means more likely defective.
        """
        raise NotImplementedError

    def predict_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Predict defect scores for a batch of images.

        Args:
            images (torch.Tensor): Batch tensor of shape ``(B, C, H, W)``.

        Returns:
            torch.Tensor: Scores tensor of shape ``(B,)`` on CPU.
        """
        raise NotImplementedError

    def state_payload(self) -> dict[str, Any]:
        """Return serializable model state for checkpointing.

        Returns:
            dict[str, Any]: State dictionary that can be saved with torch.save.
        """
        raise NotImplementedError

    def load_payload(self, payload: dict[str, Any]) -> None:
        """Load model state from a saved payload.

        Args:
            payload (dict[str, Any]): State produced by ``state_payload()``.
        """
        raise NotImplementedError
