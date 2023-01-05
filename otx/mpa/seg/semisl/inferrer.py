# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os.path as osp
from contextlib import nullcontext

import mmcv
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

from otx.mpa.modules.hooks.recording_forward_hooks import FeatureVectorHook
from otx.mpa.registry import STAGES
from otx.mpa.stage import Stage

from .stage import SemiSegStage
from otx.mpa.seg.inferrer import SegInferrer

@STAGES.register_module()
class SemiSegInferrer(SemiSegStage, SegInferrer):
    def __init__(self, **kwargs):
        SemiSegStage.__init__(self, **kwargs)

    def configure(self, model_cfg, model_ckpt, data_cfg, training=False, **kwargs):
        cfg = SemiSegStage.configure(self, model_cfg, model_ckpt, data_cfg, training=training, **kwargs)

        cfg.model.type = cfg.model.orig_type
        cfg.model.pop("orig_type", False)
        cfg.model.pop("unsup_weight", False)
        cfg.model.pop("semisl_start_iter", False)

        return cfg
