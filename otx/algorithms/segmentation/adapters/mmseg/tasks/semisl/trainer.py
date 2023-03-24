"""Train Semi-SL OTX Segmentation model with MMSEG."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.algorithms.common.adapters.mmcv.tasks.registry import STAGES
from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.segmentation.adapters.mmseg.tasks.trainer import SegTrainer

from .stage import SemiSLSegStage

logger = get_logger()


# pylint: disable=super-init-not-called
@STAGES.register_module()
class SemiSLSegTrainer(SemiSLSegStage, SegTrainer):
    """Class for semi-sl segmentation model train."""

    def __init__(self, **kwargs):
        SemiSLSegStage.__init__(self, **kwargs)
