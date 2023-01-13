# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.mpa.cls.inferrer import ClsInferrer
from otx.mpa.registry import STAGES
from otx.mpa.utils.logger import get_logger

from .stage import IncrClsStage

logger = get_logger()


@STAGES.register_module()
class IncrClsInferrer(IncrClsStage, ClsInferrer):
    def __init__(self, **kwargs):
        IncrClsStage.__init__(self, **kwargs)
