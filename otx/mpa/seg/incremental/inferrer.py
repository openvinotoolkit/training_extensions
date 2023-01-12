# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.mpa.registry import STAGES
from otx.mpa.seg.inferrer import SegInferrer

from .stage import IncrSegStage


@STAGES.register_module()
class IncrSegInferrer(IncrSegStage, SegInferrer):
    def __init__(self, **kwargs):
        IncrSegStage.__init__(self, **kwargs)
