# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
from mmcv.runner import wrap_fp16_model

from otx.mpa.deploy.utils import sync_batchnorm_2_batchnorm
from otx.mpa.exporter_mixin import ExporterMixin
from otx.mpa.registry import STAGES
from otx.mpa.seg.exporter import SegExporter
from otx.mpa.utils.logger import get_logger

from .stage import SemiSLSegStage

logger = get_logger()


@STAGES.register_module()
class SemiSLSegExporter(SemiSLSegStage, SegExporter):
    def __init__(self, **kwargs):
        SemiSLSegStage.__init__(self, **kwargs)

    def configure(self, model_cfg, model_ckpt, data_cfg, training=False, **kwargs):
        cfg = SemiSLSegStage.configure(self, model_cfg, model_ckpt, data_cfg, training=training, **kwargs)

        cfg.model.type = cfg.model.orig_type
        cfg.model.pop("orig_type", False)
        cfg.model.pop("unsup_weight", False)
        cfg.model.pop("semisl_start_iter", False)

        return cfg
