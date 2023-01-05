# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.mpa.registry import STAGES
from otx.mpa.utils.logger import get_logger
from otx.mpa.seg.trainer import SegTrainer
from .stage import SemiSegStage

logger = get_logger()


@STAGES.register_module()
class SemiSegTrainer(SemiSegStage, SegTrainer):
    def __init__(self, **kwargs):
        SemiSegStage.__init__(self, **kwargs)
