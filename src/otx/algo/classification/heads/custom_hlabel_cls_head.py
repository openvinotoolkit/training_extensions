# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for defining multi-label linear classification head."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mmengine.model import BaseModule, normal_init
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from torch import nn

if TYPE_CHECKING:
    from otx.core.data.entity.classification import HLabelInfo


@MODELS.register_module()
class CustomHierarchicalClsHead(BaseModule):
    """Custom classification head for hierarchical classification task.

    Args:
        num_multiclass_heads (int): Number of multi-class heads.
        num_multilabel_classes (int): Number of multi-label classes.
        in_channels (int): Number of channels in the input feature map.
        num_classes (int): Number of total classes.
        multiclass_loss (dict | None): Config of multi-class loss.
        multilabel_loss (dict | None): Config of multi-label loss.
        thr (float | None): Predictions with scores under the thresholds are considered
                            as negative. Defaults to 0.5.
    """

    def __init__(
        self,
        num_multiclass_heads: int,
        num_multilabel_classes: int,
        in_channels: int,
        num_classes: int,
        multiclass_loss_cfg: dict | None = None,
        multilabel_loss_cfg: dict | None = None,
        thr: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.num_multiclass_heads = num_multiclass_heads
        self.num_multilabel_classes = num_multilabel_classes
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.thr = thr

        if self.num_multiclass_heads == 0:
            msg = "num_multiclass_head should be larger than 0"
            raise ValueError(msg)

        self.multiclass_loss = MODELS.build(multiclass_loss_cfg)
        if num_multilabel_classes > 0:
            self.multilabel_loss = MODELS.build(multilabel_loss_cfg)

        self.fc = nn.Linear(self.in_channels, self.num_classes)
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize weights of the layers."""
        normal_init(self.fc, mean=0, std=0.01, bias=0)

    def pre_logits(self, feats: tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head."""
        return feats[-1]

    def forward(self, feats: tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        return self.fc(pre_logits)

    def set_hlabel_info(self, hlabel_info: HLabelInfo) -> None:
        """Set hlabel information."""
        self.hlabel_info = hlabel_info

    def _get_gt_label(self, data_samples: list[DataSample]) -> torch.Tensor:
        """Get gt labels from data samples."""
        return torch.stack([data_sample.gt_label for data_sample in data_samples])

    def _get_head_idx_to_logits_range(self, hlabel_info: HLabelInfo, idx: int) -> tuple[int, int]:
        """Get head_idx_to_logits_range information from hlabel information."""
        return (
            hlabel_info.head_idx_to_logits_range[str(idx)][0],
            hlabel_info.head_idx_to_logits_range[str(idx)][1],
        )

    def loss(self, feats: tuple[torch.Tensor], data_samples: list[DataSample], **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = {}
        cls_scores = self(feats)
        gt_labels = self._get_gt_label(data_samples)

        losses = {"loss": 0.0}

        # Multiclass loss
        num_effective_heads_in_batch = 0  # consider the label removal case
        for i in range(self.num_multiclass_heads):
            if i not in self.hlabel_info.empty_multiclass_head_indices:
                head_gt = gt_labels[:, i]
                logit_range = self._get_head_idx_to_logits_range(self.hlabel_info, i)
                head_logits = cls_scores[:, logit_range[0] : logit_range[1]]
                valid_mask = head_gt >= 0

                head_gt = head_gt[valid_mask]
                if len(head_gt) > 0:
                    head_logits = head_logits[valid_mask, :]
                    losses["loss"] += self.multiclass_loss(head_logits, head_gt)
                    num_effective_heads_in_batch += 1

        if num_effective_heads_in_batch > 0:
            losses["loss"] /= num_effective_heads_in_batch

        # Multilabel loss
        if self.num_multilabel_classes > 0:
            head_gt = gt_labels[:, self.hlabel_info.num_multiclass_heads :]
            head_logits = cls_scores[:, self.hlabel_info.num_single_label_classes :]
            valid_mask = head_gt > 0
            head_gt = head_gt[valid_mask]
            if len(head_gt) > 0:
                img_metas = [data_sample.metainfo for data_sample in data_samples]
                valid_label_mask = self.get_valid_label_mask(img_metas)
                head_logits = head_logits[valid_mask]
                losses["loss"] += self.multilabel_loss(head_logits, head_gt, valid_label_mask=valid_label_mask)

        return losses

    def predict(
        self,
        feats: tuple[torch.Tensor],
        data_samples: list[DataSample] | None = None,
    ) -> list[DataSample]:
        """Inference without augmentation.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample], optional): The annotation
                data of every samples. If not None, set ``pred_label`` of
                the input data samples. Defaults to None.

        Returns:
            List[DataSample]: A list of data samples which contains the
            predicted results.
        """
        cls_scores = self(feats)
        return self._get_predictions(cls_scores, data_samples)

    def _get_predictions(
        self,
        cls_scores: torch.Tensor,
        data_samples: list[DataSample] | None = None,
    ) -> list[DataSample]:
        """Post-process the output of head.

        Including softmax and set ``pred_label`` of data samples.
        """
        if data_samples is None:
            data_samples = [DataSample() for _ in range(cls_scores.size(0))]

        # Multiclass
        multiclass_pred_scores = []
        multiclass_pred_labels = []
        for i in range(self.num_multiclass_heads):
            logit_range = self._get_head_idx_to_logits_range(self.hlabel_info, i)
            multiclass_logit = cls_scores[:, logit_range[0] : logit_range[1]]
            multiclass_pred = torch.softmax(multiclass_logit, dim=1)
            multiclass_pred_score, multiclass_pred_label = torch.max(multiclass_pred, dim=1)

            multiclass_pred_scores.append(multiclass_pred_score.view(-1, 1))
            multiclass_pred_labels.append(multiclass_pred_label.view(-1, 1))

        multiclass_pred_scores = torch.cat(multiclass_pred_scores, dim=1)
        multiclass_pred_labels = torch.cat(multiclass_pred_labels, dim=1)

        if self.num_multilabel_classes > 0:
            multilabel_logits = cls_scores[:, self.hlabel_info.num_single_label_classes :]

            multilabel_pred_scores = torch.sigmoid(multilabel_logits)
            multilabel_pred_labels = (multilabel_pred_scores >= self.thr).int()

            pred_scores = torch.cat([multiclass_pred_scores, multilabel_pred_scores], axis=1)
            pred_labels = torch.cat([multiclass_pred_labels, multilabel_pred_labels], axis=1)
        else:
            pred_scores = multiclass_pred_scores
            pred_labels = multiclass_pred_labels

        for data_sample, score, label in zip(data_samples, pred_scores, pred_labels):
            data_sample.set_pred_score(score).set_pred_label(label)

        return data_samples

    def get_valid_label_mask(self, img_metas: list[dict]) -> list[torch.Tensor]:
        """Get valid label mask using ignored_label."""
        valid_label_mask = []
        for meta in img_metas:
            mask = torch.Tensor([1 for _ in range(self.num_classes)])
            if "ignored_labels" in meta and meta["ignored_labels"]:
                mask[meta["ignored_labels"]] = 0
            valid_label_mask.append(mask)
        return torch.stack(valid_label_mask, dim=0)
