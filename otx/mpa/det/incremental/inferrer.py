# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from contextlib import nullcontext

import torch
from mmcv.parallel import is_module_wrapper
from mmcv.runner import load_checkpoint
from mmdet.datasets import (
    ImageTilingDataset,
    build_dataloader,
    build_dataset,
    replace_ImageToTensor,
)
from mmdet.models import build_detector
from mmdet.models.detectors import TwoStageDetector
from mmdet.utils.misc import prepare_mmdet_model_for_execution

from otx.mpa.det.inferrer import DetectionInferrer
from otx.mpa.modules.hooks.recording_forward_hooks import (
    ActivationMapHook,
    DetSaliencyMapHook,
    FeatureVectorHook,
)
from otx.mpa.registry import STAGES
from otx.mpa.utils.logger import get_logger

from .stage import IncrDetectionStage

logger = get_logger()


@STAGES.register_module()
class IncrDetectionInferrer(IncrDetectionStage, DetectionInferrer):
    def __init__(self, **kwargs):
        IncrDetectionStage.__init__(self, **kwargs)
