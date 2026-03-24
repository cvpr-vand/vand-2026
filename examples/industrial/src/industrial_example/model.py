# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Baseline ResNet-18 feature-memory model for industrial anomaly detection."""

# pyright: reportMissingImports=false

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import feature_extraction


class BaselineModel(nn.Module):
    """Feature-memory baseline model using a pretrained ResNet-18 backbone.

    Args:
        max_features (int): Maximum number of stored patch features. Defaults to 50000.
        device (str | None): Execution device. If None, uses CUDA when available.
    """

    def __init__(self, max_features: int = 50000, device: str | None = None) -> None:
        super().__init__()

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        weights = models.ResNet18_Weights.DEFAULT
        backbone = models.resnet18(weights=weights)
        self.extractor = feature_extraction.create_feature_extractor(backbone, {"layer2": "feat"})
        self.extractor.eval()
        for param in self.extractor.parameters():
            param.requires_grad = False

        self.max_features = int(max_features)
        self.register_buffer("feature_bank", torch.empty((0, 128), dtype=torch.float32))
        self.pool = nn.AdaptiveAvgPool2d((64, 64))
        self.to(self.device)

    def _extract_feature_map(self, x: Any) -> Any:
        """Extract intermediate feature map from input tensor."""

        with torch.no_grad():
            out = self.extractor(x)["feat"]
        return out

    def _flatten_features(self, feat_map: Any) -> Any:
        """Flatten and normalize spatial feature vectors."""

        b, c, h, w = feat_map.shape
        flat = feat_map.permute(0, 2, 3, 1).reshape(b * h * w, c)
        return F.normalize(flat, dim=1)

    def fit(self, train_dataloader: Any) -> None:
        """Build the feature bank from training images.

        Args:
            train_dataloader (Any): Dataloader that yields training batches.

        Raises:
            ValueError: If the dataloader yields no batches.
        """

        self.eval()
        chunks: list[Any] = []
        for batch in train_dataloader:
            images = batch.image.to(self.device, non_blocking=True)
            feat = self._extract_feature_map(images)
            flat = self._flatten_features(feat)
            chunks.append(flat.cpu())

        if len(chunks) == 0:
            raise ValueError("Training dataloader is empty")

        bank = torch.cat(chunks, dim=0)
        if bank.shape[0] > self.max_features:
            idx = torch.randperm(bank.shape[0])[: self.max_features]
            bank = bank[idx]
        self.feature_bank = bank.to(self.device)

    def predict(self, image: Any) -> tuple[Any, Any]:
        """Predict anomaly map and scalar score for one image tensor.

        Args:
            image (Any): Input image tensor, shape (C,H,W) or (1,C,H,W).

        Returns:
            tuple[Any, Any]: Anomaly map tensor and anomaly score tensor on CPU.

        Raises:
            RuntimeError: If the model is not fitted.
            ValueError: If more than one image is passed.
        """

        if self.feature_bank.numel() == 0:
            raise RuntimeError("Model is not fitted; feature bank is empty")

        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)

        with torch.no_grad():
            feat_map = self._extract_feature_map(image)
            b, _, h, w = feat_map.shape
            if b != 1:
                raise ValueError("predict expects a single image tensor")
            flat = self._flatten_features(feat_map)
            distances = torch.cdist(flat, self.feature_bank)
            patch_scores = distances.min(dim=1).values.reshape(1, 1, h, w)
            up = F.interpolate(
                patch_scores,
                size=(image.shape[-2], image.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            anomaly_map = up.squeeze(0).squeeze(0)
            anomaly_score = anomaly_map.max()
            return anomaly_map.detach().cpu(), anomaly_score.detach().cpu()

    def get_threshold(self, val_dataloader: Any) -> float:
        """Estimate anomaly threshold from validation data.

        Args:
            val_dataloader (Any): Dataloader that yields validation batches.

        Returns:
            float: Threshold computed as mean + 3 * std of validation scores.

        Raises:
            ValueError: If the dataloader yields no samples.
        """

        scores: list[float] = []
        for batch in val_dataloader:
            images = batch.image
            batch_size = int(images.shape[0])
            for idx in range(batch_size):
                _, score = self.predict(images[idx])
                scores.append(float(score.item()))
        if len(scores) == 0:
            raise ValueError("Validation dataloader is empty")
        arr = torch.tensor(scores, dtype=torch.float32)
        return float((arr.mean() + 3.0 * arr.std(unbiased=False)).item())

    def save(self, path: str | Path) -> None:
        """Save model state to disk.

        Args:
            path (str | Path): Output path for checkpoint file.
        """

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "max_features": self.max_features,
            "feature_bank": self.feature_bank.detach().cpu(),
        }
        torch.save(state, output)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> BaselineModel:
        """Load model state from disk.

        Args:
            path (str | Path): Checkpoint path.
            device (str | None): Optional execution device override.

        Returns:
            BaselineModel: Loaded model instance.
        """

        ckpt = torch.load(Path(path), map_location="cpu", weights_only=True)
        model = cls(max_features=int(ckpt["max_features"]), device=device)
        model.feature_bank = ckpt["feature_bank"].to(model.device)
        model.eval()
        return model
