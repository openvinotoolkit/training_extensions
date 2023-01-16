# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.mpa.det.trainer import DetectionTrainer
from otx.mpa.registry import STAGES
from otx.mpa.utils.logger import get_logger

from .stage import IncrDetectionStage

logger = get_logger()


@STAGES.register_module()
class IncrDetectionTrainer(IncrDetectionStage, DetectionTrainer):
    def __init__(self, **kwargs):
        IncrDetectionStage.__init__(self, **kwargs)
