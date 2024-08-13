# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Classifier for H-Label Classification."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .base_classifier import ImageClassifier

if TYPE_CHECKING:
    from torch import nn


class HLabelClassifier(ImageClassifier):
    """HLabel Classifier."""

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module | None,
        head: nn.Module,
        init_cfg: dict | list[dict] | None = None,
    ):
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            loss=head.multiclass_loss,
            init_cfg=init_cfg,
        )

    def loss(self, inputs: torch.Tensor, labels: torch.Tensor, **kwargs) -> torch.Tensor:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            labels (torch.Tensor): The annotation data of
                every samples.

        Returns:
            torch.Tensor: loss components
        """
        feats = self.extract_feat(inputs)
        return self.head.loss(feats, labels, **kwargs)

    @torch.no_grad()
    def _forward_explain(self, images: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        from otx.algo.explain.explain_algo import feature_vector_fn

        x = self.backbone(images)
        backbone_feat = x

        feature_vector = feature_vector_fn(backbone_feat)
        saliency_map = self.explainer.func(backbone_feat)

        if hasattr(self, "neck") and self.neck is not None:
            x = self.neck(x)

        logits = self.head(x)
        pred_results = self.head._get_predictions(logits)  # noqa: SLF001
        scores = pred_results["scores"]
        preds = pred_results["labels"]

        outputs = {
            "logits": logits,
            "feature_vector": feature_vector,
            "saliency_map": saliency_map,
        }

        if not torch.jit.is_tracing():
            outputs["scores"] = scores
            outputs["preds"] = preds

        return outputs
