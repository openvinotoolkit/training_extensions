# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.algorithms.common.utils.logger import get_logger
from otx.mpa.cls.trainer import ClsTrainer
from otx.mpa.registry import STAGES

from .stage import SemiSLClsStage

logger = get_logger()


@STAGES.register_module()
class SemiSLClsTrainer(SemiSLClsStage, ClsTrainer):
    def __init__(self, **kwargs):
        SemiSLClsStage.__init__(self, **kwargs)
