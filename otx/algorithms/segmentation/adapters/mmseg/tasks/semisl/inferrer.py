"""Inferenc for Semi-SL OTX classification with MMCLS."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.algorithms.common.adapters.mmcv.tasks.registry import STAGES
from otx.algorithms.segmentation.adapters.mmseg.tasks.inferrer import SegInferrer

from .stage import SemiSLSegStage


# pylint: disable=super-init-not-called
@STAGES.register_module()
class SemiSLSegInferrer(SemiSLSegStage, SegInferrer):
    """Inference class for Semi-SL."""

    def __init__(self, **kwargs):
        SemiSLSegStage.__init__(self, **kwargs)

    def configure(self, model_cfg, model_ckpt, data_cfg, training=False, **kwargs):
        """Patch config for semi-sl classification."""
        cfg = SemiSLSegStage.configure(self, model_cfg, model_ckpt, data_cfg, training=training, **kwargs)

        cfg.model.type = cfg.model.orig_type
        cfg.model.pop("orig_type", False)
        cfg.model.pop("unsup_weight", False)
        cfg.model.pop("semisl_start_iter", False)

        return cfg
