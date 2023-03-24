"""Inference for OTX segmentation model with Incremental learning."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.algorithms.common.adapters.mmcv.tasks.registry import STAGES
from otx.algorithms.segmentation.adapters.mmseg.tasks.inferrer import SegInferrer

from .stage import IncrSegStage


# pylint: disable=super-init-not-called
@STAGES.register_module()
class IncrSegInferrer(IncrSegStage, SegInferrer):
    """Inference class for incremental learning."""

    def __init__(self, **kwargs):
        IncrSegStage.__init__(self, **kwargs)
