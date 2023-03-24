"""Semi-SL Object detection Task with MMDET."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.algorithms.common.adapters.mmcv.tasks.registry import STAGES
from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.detection.adapters.mmdet.tasks.trainer import DetectionTrainer

from .stage import SemiSLDetectionStage

logger = get_logger()


# pylint: disable=super-init-not-called
@STAGES.register_module()
class SemiSLDetectionTrainer(SemiSLDetectionStage, DetectionTrainer):
    """Train class for semi-sl object detection."""

    def __init__(self, **kwargs):
        SemiSLDetectionStage.__init__(self, **kwargs)
