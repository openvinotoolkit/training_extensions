# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Classifier for H-Label Classification."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import torch

from otx.algo.classification.heads.hlabel_cls_head import HierarchicalClsHead
from otx.algo.classification.utils.ignored_labels import get_valid_label_mask

from .base_classifier import ImageClassifier

if TYPE_CHECKING:
    from torch import nn


class HLabelClassifier(ImageClassifier):
    """Hierarchical label classifier.

    Args:
        backbone (nn.Module): Backbone network.
        neck (nn.Module | None): Neck network.
        head (nn.Module): Head network.
        multiclass_loss (nn.Module): Multiclass loss function.
        multilabel_loss (nn.Module | None, optional): Multilabel loss function.
        init_cfg (dict | list[dict] | None, optional): Initialization configuration.

    Attributes:
        multiclass_loss (nn.Module): Multiclass loss function.
        multilabel_loss (nn.Module | None): Multilabel loss function.
        is_ignored_label_loss (bool): Flag indicating if ignored label loss is used.

    Methods:
        loss(inputs, labels, **kwargs): Calculate losses from a batch of inputs and data samples.
        _forward_explain(images): Perform forward pass for explanation.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module | None,
        head: HierarchicalClsHead,
        multiclass_loss: nn.Module,
        multilabel_loss: nn.Module | None = None,
        init_cfg: dict | list[dict] | None = None,
    ):
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            loss=multiclass_loss,
            init_cfg=init_cfg,
        )

        self.multiclass_loss = multiclass_loss
        self.multilabel_loss = None
        if self.head.num_multilabel_classes > 0 and multilabel_loss is not None:
            self.multilabel_loss = multilabel_loss
            self.is_ignored_label_loss = "valid_label_mask" in inspect.getfullargspec(self.multilabel_loss.forward).args

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
        cls_scores = self.extract_feat(inputs, stage="head")
        loss_score = torch.tensor(0.0, device=cls_scores.device)

        # Multiclass loss
        num_effective_heads_in_batch = 0  # consider the label removal case
        for i in range(self.head.num_multiclass_heads):
            if i not in self.head.empty_multiclass_head_indices:
                head_gt = labels[:, i]
                logit_range = self.head._get_head_idx_to_logits_range(i)  # noqa: SLF001
                head_logits = cls_scores[:, logit_range[0] : logit_range[1]]
                valid_mask = head_gt >= 0

                head_gt = head_gt[valid_mask]
                if len(head_gt) > 0:
                    head_logits = head_logits[valid_mask, :]
                    loss_score += self.multiclass_loss(head_logits, head_gt)
                    num_effective_heads_in_batch += 1

        if num_effective_heads_in_batch > 0:
            loss_score /= num_effective_heads_in_batch

        # Multilabel loss
        if self.head.num_multilabel_classes > 0:
            head_gt = labels[:, self.head.num_multiclass_heads :]
            head_logits = cls_scores[:, self.head.num_single_label_classes :]
            valid_mask = head_gt > 0
            head_gt = head_gt[valid_mask]
            if len(head_gt) > 0 and self.multilabel_loss is not None:
                head_logits = head_logits[valid_mask]
                imgs_info = kwargs.pop("imgs_info", None)
                if imgs_info is not None and self.is_ignored_label_loss:
                    valid_label_mask = get_valid_label_mask(imgs_info, self.head.num_classes).to(head_logits.device)
                    valid_label_mask = valid_label_mask[:, self.head.num_single_label_classes :]
                    valid_label_mask = valid_label_mask[valid_mask]
                    kwargs["valid_label_mask"] = valid_label_mask
                loss_score += self.multilabel_loss(head_logits, head_gt, **kwargs)

        return loss_score

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
