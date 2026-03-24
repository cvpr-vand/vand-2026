# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Baseline model for the retail defect detection track."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision.models as models


class BaselineModel(nn.Module):
    """ResNet-18 feature-based baseline model.

    Args:
        device (str | Any | None): Device override. Uses CUDA when available if
            None is provided.
    """

    def __init__(self, device: str | Any | None = None) -> None:
        super().__init__()
        self.device = (
            torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        try:
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception:
            backbone = models.resnet18(weights=None)

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()

        self.register_buffer("mean_no_defect", torch.zeros(512))
        self.register_buffer("mean_defect", torch.zeros(512))
        self.register_buffer("threshold", torch.tensor(0.5))

        self.reference_embeddings: dict[str, list[Any]] = {}
        self.to(self.device)

    def _encode(self, images: Any) -> Any:
        """Encode images into normalized feature embeddings."""
        if images.ndim == 3:
            images = images.unsqueeze(0)
        with torch.no_grad():
            feats = self.feature_extractor(images.to(self.device))
            feats = feats.flatten(1)
            feats = f.normalize(feats, p=2, dim=1)
        return feats

    def _center_score(self, embeddings: Any) -> Any:
        """Score embeddings by cosine distance to normal center."""
        no_defect_norm = torch.norm(self.mean_no_defect).item()
        if no_defect_norm < 1e-8:
            return torch.full((embeddings.shape[0],), 0.5, device=embeddings.device)

        center = f.normalize(self.mean_no_defect.unsqueeze(0), p=2, dim=1)
        cosine = f.cosine_similarity(embeddings, center.expand_as(embeddings), dim=1)
        distance = 1.0 - cosine
        logits = (distance - self.threshold) * 8.0
        return torch.sigmoid(logits)

    def _prepare_reference_embeddings(self, refs: list[Any]) -> Any:
        """Convert reference tensors into normalized embeddings."""
        vectors: list[Any] = []
        for ref in refs:
            if ref.ndim == 1:
                vectors.append(f.normalize(ref.to(self.device), p=2, dim=0))
            else:
                encoded = self._encode(ref)
                vectors.append(encoded.squeeze(0))
        if not vectors:
            raise ValueError("reference_embeddings is empty")
        return torch.stack(vectors, dim=0)

    def fit(self, train_dataloader: Any, reference_fn: Any = None) -> BaselineModel:
        """Fit baseline centers and optional cached reference embeddings.

        Args:
            train_dataloader (Any): Dataloader yielding training batches.
            reference_fn (Any): Callable returning references for item IDs.

        Returns:
            BaselineModel: Fitted model instance.
        """
        no_defect_embeddings: list[Any] = []
        defect_embeddings: list[Any] = []
        no_defect_scores: list[float] = []
        defect_scores: list[float] = []

        if reference_fn is not None:
            self.reference_embeddings = defaultdict(list)

        for batch in train_dataloader:
            images = batch.image
            labels = batch.label
            item_ids = batch.item_identifier

            emb = self._encode(images)

            if labels is not None:
                labels_t = labels.to(self.device)
                for i in range(emb.shape[0]):
                    if int(labels_t[i].item()) == 1:
                        defect_embeddings.append(emb[i].detach().cpu())
                    else:
                        no_defect_embeddings.append(emb[i].detach().cpu())

            if reference_fn is not None and item_ids is not None:
                unique_items = {str(item_id) for item_id in item_ids}
                for item_id in unique_items:
                    if item_id in self.reference_embeddings:
                        continue
                    refs = reference_fn(item_id)
                    if not refs:
                        continue
                    ref_emb = self._prepare_reference_embeddings(refs)
                    self.reference_embeddings[item_id] = [v.detach().cpu() for v in ref_emb]

        if no_defect_embeddings:
            self.mean_no_defect.copy_(
                f.normalize(
                    torch.stack(no_defect_embeddings).mean(dim=0).to(self.device),
                    p=2,
                    dim=0,
                )
            )
        if defect_embeddings:
            self.mean_defect.copy_(
                f.normalize(
                    torch.stack(defect_embeddings).mean(dim=0).to(self.device),
                    p=2,
                    dim=0,
                )
            )

        if no_defect_embeddings:
            no_stack = torch.stack(no_defect_embeddings).to(self.device)
            no_defect_scores = self._center_score(no_stack).detach().cpu().tolist()
        if defect_embeddings:
            d_stack = torch.stack(defect_embeddings).to(self.device)
            defect_scores = self._center_score(d_stack).detach().cpu().tolist()

        if no_defect_scores and defect_scores:
            threshold = 0.5 * (sum(no_defect_scores) / len(no_defect_scores) + sum(defect_scores) / len(defect_scores))
            self.threshold.copy_(torch.tensor(float(min(max(threshold, 0.0), 1.0)), device=self.device))

        return self

    def predict(
        self,
        image: Any,
        reference_embeddings: list[Any] | None = None,
    ) -> float:
        """Predict a defect score for a single image.

        Args:
            image (Any): Input image tensor.
            reference_embeddings (list[Any] | None): Optional reference embeddings.

        Returns:
            float: Defect score in [0, 1].
        """
        embedding = self._encode(image).squeeze(0)

        if reference_embeddings:
            refs = self._prepare_reference_embeddings(reference_embeddings)
            query = embedding.unsqueeze(0).expand(refs.shape[0], -1)
            similarity = f.cosine_similarity(query, refs, dim=1).max()
            score = (1.0 - similarity).clamp(0.0, 1.0)
            return float(score.detach().cpu().item())

        score = self._center_score(embedding.unsqueeze(0)).squeeze(0)
        return float(score.detach().cpu().item())

    def predict_batch(self, images: Any) -> Any:
        """Predict defect scores for a batch of images.

        Args:
            images (Any): Batch tensor of images.

        Returns:
            Any: Batch of scores on CPU.
        """
        embeddings = self._encode(images)
        return self._center_score(embeddings).detach().cpu()

    def state_payload(self) -> dict[str, Any]:
        """Return serializable model and reference state.

        Returns:
            dict[str, Any]: State payload for checkpointing.
        """
        return {
            "state_dict": self.state_dict(),
            "reference_embeddings": {
                key: [tensor.clone().cpu() for tensor in values] for key, values in self.reference_embeddings.items()
            },
        }

    def load_payload(self, payload: dict[str, Any]) -> None:
        """Load model and reference state from a payload.

        Args:
            payload (dict[str, Any]): Payload produced by state_payload().
        """
        self.load_state_dict(payload["state_dict"])
        refs = payload.get("reference_embeddings", {})
        self.reference_embeddings = {
            str(key): [tensor.to(torch.float32).cpu() for tensor in value] for key, value in refs.items()
        }
