# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.mpa.registry import STAGES
from otx.mpa.seg.inferrer import SegInferrer

from .stage import SemiSLSegStage


@STAGES.register_module()
class SemiSLSegInferrer(SemiSLSegStage, SegInferrer):
    def __init__(self, **kwargs):
        SemiSLSegStage.__init__(self, **kwargs)

    def configure(self, model_cfg, model_ckpt, data_cfg, training=False, **kwargs):
        cfg = SemiSLSegStage.configure(self, model_cfg, model_ckpt, data_cfg, training=training, **kwargs)

        cfg.model.type = cfg.model.orig_type
        cfg.model.pop("orig_type", False)
        cfg.model.pop("unsup_weight", False)
        cfg.model.pop("semisl_start_iter", False)

        return cfg
