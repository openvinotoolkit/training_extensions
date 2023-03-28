"""Semi-SL Trainer for OTX Classification with MMCLS."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.algorithms.classification.adapters.mmcls.tasks.trainer import ClsTrainer
from otx.algorithms.common.adapters.mmcv.tasks.registry import STAGES
from otx.algorithms.common.utils.logger import get_logger

from .stage import SemiSLClsStage

logger = get_logger()


# pylint: disable=super-init-not-called
@STAGES.register_module()
class SemiSLClsTrainer(SemiSLClsStage, ClsTrainer):
    """Trainer class for Semi-SL."""

    def __init__(self, **kwargs):
        SemiSLClsStage.__init__(self, **kwargs)
