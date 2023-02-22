# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
from mmcv.runner import wrap_fp16_model

from otx.mpa.deploy.utils import sync_batchnorm_2_batchnorm
from otx.mpa.det.exporter import DetectionExporter
from otx.mpa.exporter_mixin import ExporterMixin
from otx.mpa.registry import STAGES
from otx.mpa.utils.logger import get_logger

from .stage import SemiSLDetectionStage

logger = get_logger()


@STAGES.register_module()
class SemiSLDetectionExporter(SemiSLDetectionStage, DetectionExporter):
    def __init__(self, **kwargs):
        SemiSLDetectionStage.__init__(self, **kwargs)

    def _get_feature_module(self, model):
        model = super()._get_feature_module(model)
        return model.model_t
