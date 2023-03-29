"""Incremental learning trainer for OTX Detection with MMDET."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.algorithms.classification.adapters.mmcls.tasks.trainer import ClsTrainer
from otx.algorithms.common.adapters.mmcv.tasks.registry import STAGES
from otx.algorithms.common.utils.logger import get_logger

from .stage import IncrClsStage

logger = get_logger()


# pylint: disable=super-init-not-called
@STAGES.register_module()
class IncrClsTrainer(IncrClsStage, ClsTrainer):
    """Trainer for incremental classification."""

    def __init__(self, **kwargs):
        IncrClsStage.__init__(self, **kwargs)
