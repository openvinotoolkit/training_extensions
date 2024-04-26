# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This class and its supporting functions are adapted from the mmdet.
# Please refer to https://github.com/open-mmlab/mmdetection/
"""MMDet MaskRCNN."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .two_stage import TwoStageDetector

if TYPE_CHECKING:
    import torch
    from mmdet.structures.det_data_sample import DetDataSample
    from mmengine.config import ConfigDict


class MaskRCNN(TwoStageDetector):
    """Implementation of `Mask R-CNN <https://arxiv.org/abs/1703.06870>`."""

    def __init__(
        self,
        backbone: ConfigDict,
        rpn_head: ConfigDict,
        roi_head: ConfigDict,
        train_cfg: ConfigDict,
        test_cfg: ConfigDict,
        neck: ConfigDict | dict | None = None,
        data_preprocessor: ConfigDict | dict | None = None,
        init_cfg: ConfigDict | dict | list[ConfigDict | dict] | None = None,
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
            data_preprocessor=data_preprocessor,
        )

    def export(
        self,
        batch_inputs: torch.Tensor,
        data_samples: list[DetDataSample],
    ) -> tuple[torch.Tensor, ...]:
        """Export MaskRCNN detector."""
        x = self.extract_feat(batch_inputs)

        rpn_results_list = self.rpn_head.export(
            x,
            data_samples,
            rescale=False,
        )

        return self.roi_head.export(
            x,
            rpn_results_list,
            data_samples,
            rescale=False,
        )
