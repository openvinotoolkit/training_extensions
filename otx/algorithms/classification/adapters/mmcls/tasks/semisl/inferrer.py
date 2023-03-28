"""Inference task for Semi-SL OTX classification with MMCLS."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.algorithms.classification.adapters.mmcls.tasks.inferrer import ClsInferrer
from otx.algorithms.common.adapters.mmcv.tasks.registry import STAGES
from otx.algorithms.common.utils.logger import get_logger

from .stage import SemiSLClsStage

logger = get_logger()


# pylint: disable=super-init-not-called
@STAGES.register_module()
class SemiSLClsInferrer(SemiSLClsStage, ClsInferrer):
    """Semi-SL Inferencer."""

    def __init__(self, **kwargs):
        SemiSLClsStage.__init__(self, **kwargs)
