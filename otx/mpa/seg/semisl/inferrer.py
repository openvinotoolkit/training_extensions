# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.mpa.registry import STAGES
from otx.mpa.seg.inferrer import SegInferrer

from .stage import SemiSegStage


@STAGES.register_module()
class SemiSegInferrer(SemiSegStage, SegInferrer):
    def __init__(self, **kwargs):
        SemiSegStage.__init__(self, **kwargs)

    def configure(self, model_cfg, model_ckpt, data_cfg, training=False, **kwargs):
        cfg = SemiSegStage.configure(self, model_cfg, model_ckpt, data_cfg, training=training, **kwargs)

        cfg.model.type = cfg.model.orig_type
        cfg.model.pop("orig_type", False)
        cfg.model.pop("unsup_weight", False)
        cfg.model.pop("semisl_start_iter", False)

        return cfg
