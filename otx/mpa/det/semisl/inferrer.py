# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.algorithms.common.utils.logger import get_logger
from otx.mpa.det.inferrer import DetectionInferrer
from otx.mpa.registry import STAGES

from .stage import SemiSLDetectionStage

logger = get_logger()


@STAGES.register_module()
class SemiSLDetectionInferrer(SemiSLDetectionStage, DetectionInferrer):
    def __init__(self, **kwargs):
        SemiSLDetectionStage.__init__(self, **kwargs)

    def _get_feature_module(self, model):
        model = super()._get_feature_module(model)
        return model.model_t
