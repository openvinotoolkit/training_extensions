# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from typing import List, Tuple

import torch
from mmcv.parallel import MMDataParallel, is_module_wrapper
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.models import build_detector
from mmdet.parallel import MMDataCPU
from mmdet.utils.deployment import get_feature_vector, get_saliency_map

from otx.mpa.det.inferrer import DetectionInferrer
from otx.mpa.det.semisl import SemiSLDetectionStage
from otx.mpa.registry import STAGES
from otx.mpa.utils.logger import get_logger

logger = get_logger()


@STAGES.register_module()
class SemiSLDetectionInferrer(SemiSLDetectionStage, DetectionInferrer):
    def __init__(self, **kwargs):
        SemiSLDetectionStage.__init__(self, **kwargs)

    def _get_feature_module(self, eval_model):
        return eval_model.module.model_t
