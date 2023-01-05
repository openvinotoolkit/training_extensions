# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os.path as osp
from contextlib import nullcontext

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.parallel import MMDataCPU

from otx.mpa.modules.hooks.recording_forward_hooks import FeatureVectorHook
from otx.mpa.registry import STAGES
from otx.mpa.stage import Stage

from .stage import IncrSegStage
from otx.mpa.seg.inferrer import SegInferrer


@STAGES.register_module()
class IncrSegInferrer(IncrSegStage, SegInferrer):
    def __init__(self, **kwargs):
        IncrSegStage.__init__(self, **kwargs)

