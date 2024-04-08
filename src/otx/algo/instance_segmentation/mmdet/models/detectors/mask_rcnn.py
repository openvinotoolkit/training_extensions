"""The original source code is from mmdet. Please refer to https://github.com/open-mmlab/mmdetection/."""
# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING

from mmengine.config import ConfigDict
from mmengine.registry import MODELS

from .two_stage import TwoStageDetector

if TYPE_CHECKING:
    from otx.algo.instance_segmentation.mmdet.models.utils import OptConfigType, OptMultiConfig


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
        neck: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
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
