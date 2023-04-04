"""Trainer for Incremental OTX Segmentation with MMSEG."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.algorithms.common.adapters.mmcv.tasks.registry import STAGES
from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.segmentation.adapters.mmseg.tasks.trainer import SegTrainer

from .stage import IncrSegStage

logger = get_logger()


# pylint: disable=super-init-not-called
@STAGES.register_module()
class IncrSegTrainer(IncrSegStage, SegTrainer):
    """Trainer for incremental segmentation."""

    def __init__(self, **kwargs):
        IncrSegStage.__init__(self, **kwargs)
