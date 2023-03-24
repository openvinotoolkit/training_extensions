"""Inference Incremental learning model of OTX detection."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.algorithms.common.adapters.mmcv.tasks.registry import STAGES
from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.detection.adapters.mmdet.tasks.inferrer import DetectionInferrer

from .stage import IncrDetectionStage

logger = get_logger()


# pylint: disable=super-init-not-called
@STAGES.register_module()
class IncrDetectionInferrer(IncrDetectionStage, DetectionInferrer):
    """Inferencer for OTX Detection incremental learngin with MMDET."""

    def __init__(self, **kwargs):
        IncrDetectionStage.__init__(self, **kwargs)
