# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import glob
import numbers
import os
import time

import mmcv
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from mmcv import get_git_hash
from mmseg import __version__
from mmseg.models import build_segmentor
from mmseg.utils import collect_env

from otx.mpa.registry import STAGES
from otx.mpa.utils.logger import get_logger

from ..builder import build_dataset
from ..train import train_segmentor
from .stage import SemiSegStage
from otx.mpa.seg.trainer import SegTrainer


logger = get_logger()


@STAGES.register_module()
class SemiSegTrainer(SemiSegStage, SegTrainer):
    def __init__(self, **kwargs):
        SemiSegStage.__init__(self, **kwargs)
