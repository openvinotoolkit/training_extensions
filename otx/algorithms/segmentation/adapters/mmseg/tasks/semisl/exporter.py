"""Export task for Semi-SL OTX Segmentation with MMSEG."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.algorithms.common.adapters.mmcv.tasks.registry import STAGES
from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.segmentation.adapters.mmseg.tasks.exporter import SegExporter

from .stage import SemiSLSegStage

logger = get_logger()


@STAGES.register_module()
class SemiSLSegExporter(SemiSLSegStage, SegExporter):
    """Exporter for semi-sl segmentation."""

    def __init__(self, **kwargs):
        SemiSLSegStage.__init__(self, **kwargs)

    def configure(self, model_cfg, model_ckpt, data_cfg, training=False, **kwargs):
        """Patch config for semi-sl segmentation."""
        cfg = SemiSLSegStage.configure(self, model_cfg, model_ckpt, data_cfg, training=training, **kwargs)

        cfg.model.type = cfg.model.orig_type
        cfg.model.pop("orig_type", False)
        cfg.model.pop("unsup_weight", False)
        cfg.model.pop("semisl_start_iter", False)

        return cfg
