"""Train task for Incremental Learning for OTX Detection with MMDET."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.algorithms.common.adapters.mmcv.tasks.registry import STAGES
from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.detection.adapters.mmdet.tasks.trainer import DetectionTrainer

from .stage import IncrDetectionStage

logger = get_logger()


# pylint: disable=super-init-not-called
@STAGES.register_module()
class IncrDetectionTrainer(IncrDetectionStage, DetectionTrainer):
    """Train class for incremental object detection."""

    def __init__(self, **kwargs):
        IncrDetectionStage.__init__(self, **kwargs)
