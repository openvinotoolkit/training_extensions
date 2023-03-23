# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.algorithms.common.utils.logger import get_logger
from otx.mpa.cls.trainer import ClsTrainer
from otx.mpa.registry import STAGES

from .stage import IncrClsStage

logger = get_logger()


@STAGES.register_module()
class IncrClsTrainer(IncrClsStage, ClsTrainer):
    def __init__(self, **kwargs):
        IncrClsStage.__init__(self, **kwargs)
