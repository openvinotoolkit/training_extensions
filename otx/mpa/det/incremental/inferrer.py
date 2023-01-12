# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.mpa.det.inferrer import DetectionInferrer
from otx.mpa.registry import STAGES
from otx.mpa.utils.logger import get_logger

from .stage import IncrDetectionStage

logger = get_logger()


@STAGES.register_module()
class IncrDetectionInferrer(IncrDetectionStage, DetectionInferrer):
    def __init__(self, **kwargs):
        IncrDetectionStage.__init__(self, **kwargs)
