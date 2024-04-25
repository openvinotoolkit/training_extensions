# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This class and its supporting functions are adapted from the mmdet.
# Please refer to https://github.com/open-mmlab/mmdetection/
"""MMDet MaskRCNN."""
from __future__ import annotations

from typing import TYPE_CHECKING

from mmengine.registry import MODELS

from .two_stage import TwoStageDetector

if TYPE_CHECKING:
    from mmengine.config import ConfigDict


@MODELS.register_module()
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
