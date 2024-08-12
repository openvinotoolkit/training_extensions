# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Classifier for H-Label Classification."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base_classifier import ImageClassifier

if TYPE_CHECKING:
    import torch
    from torch import nn


class HLabelClassifier(ImageClassifier):
    """HLabel Classifier."""

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module | None,
        head: nn.Module,
        optimize_gap: bool = True,
        init_cfg: dict | list[dict] | None = None,
    ):
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            loss=head.multiclass_loss,
            optimize_gap=optimize_gap,
            init_cfg=init_cfg,
        )

    def loss(self, inputs: torch.Tensor, labels: torch.Tensor, **kwargs) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            labels (torch.Tensor): The annotation data of
                every samples.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        feats = self.extract_feat(inputs)
        return self.head.loss(feats, labels, **kwargs)
