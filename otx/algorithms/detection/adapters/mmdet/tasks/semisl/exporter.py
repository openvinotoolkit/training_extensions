"""Exporter for Semi-SL Object Detection."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.algorithms.common.adapters.mmcv.tasks.registry import STAGES
from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.detection.adapters.mmdet.tasks.exporter import DetectionExporter

from .stage import SemiSLDetectionStage

logger = get_logger()


@STAGES.register_module()
class SemiSLDetectionExporter(SemiSLDetectionStage, DetectionExporter):
    """Exporter class for Semi-SL Object detection."""

    def __init__(self, **kwargs):
        SemiSLDetectionStage.__init__(self, **kwargs)

    def _get_feature_module(self, model):
        model = super()._get_feature_module(model)
        return model.model_t
