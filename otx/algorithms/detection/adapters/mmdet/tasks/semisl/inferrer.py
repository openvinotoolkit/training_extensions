"""Inference task for Semi-SL OTX Detection with MMDET."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.algorithms.common.adapters.mmcv.tasks.registry import STAGES
from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.detection.adapters.mmdet.tasks.inferrer import DetectionInferrer

from .stage import SemiSLDetectionStage

logger = get_logger()


# pylint: disable=super-init-not-called
@STAGES.register_module()
class SemiSLDetectionInferrer(SemiSLDetectionStage, DetectionInferrer):
    """Class for semi-sl detection."""

    def __init__(self, **kwargs):
        SemiSLDetectionStage.__init__(self, **kwargs)

    def _get_feature_module(self, model):
        model = super()._get_feature_module(model)
        return model.model_t
