# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This class and its supporting functions are adapted from the mmdet.
# Please refer to https://github.com/open-mmlab/mmdetection/
"""MMDet MaskRCNN."""

from __future__ import annotations

from typing import TYPE_CHECKING

from otx.algo.instance_segmentation.two_stage import TwoStageDetector

if TYPE_CHECKING:
    import torch
    from omegaconf import DictConfig
    from torch import nn


class MaskRCNN(TwoStageDetector):
    """Implementation of `Mask R-CNN <https://arxiv.org/abs/1703.06870>`."""

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        rpn_head: nn.Module,
        roi_head: nn.Module,
        train_cfg: DictConfig,
        test_cfg: DictConfig,
        init_cfg: DictConfig | dict | list[DictConfig | dict] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
        )

    def export(
        self,
        batch_inputs: torch.Tensor,
        batch_img_metas: list[dict],
    ) -> tuple[torch.Tensor, ...]:
        """Export MaskRCNN detector."""
        x = self.extract_feat(batch_inputs)

        rpn_results_list = self.rpn_head.export(
            x,
            batch_img_metas,
            rescale=False,
        )

        return self.roi_head.export(
            x,
            rpn_results_list,
            batch_img_metas,
            rescale=False,
        )
