# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.mpa.registry import STAGES
from otx.mpa.seg.trainer import SegTrainer
from otx.mpa.utils.logger import get_logger

from .stage import IncrSegStage

logger = get_logger()


@STAGES.register_module()
class IncrSegTrainer(IncrSegStage, SegTrainer):
    def __init__(self, **kwargs):
        IncrSegStage.__init__(self, **kwargs)
