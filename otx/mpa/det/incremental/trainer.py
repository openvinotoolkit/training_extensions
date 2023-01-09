# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import glob
import os.path as osp
import time

from mmcv.utils import get_git_hash
from mmdet import __version__
from mmdet.apis import train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env
from torch import nn

from otx.mpa.det.incremental import IncrDetectionStage
from otx.mpa.det.trainer import DetectionTrainer
from otx.mpa.modules.utils.task_adapt import extract_anchor_ratio
from otx.mpa.registry import STAGES
from otx.mpa.utils.logger import get_logger

# TODO[JAEGUK]: Remove import detection_tasks
# from detection_tasks.apis.detection.config_utils import cluster_anchors


logger = get_logger()


# FIXME DetectionTrainer does not inherit from stage
@STAGES.register_module()
class IncrDetectionTrainer(IncrDetectionStage, DetectionTrainer):
    def __init__(self, **kwargs):
        IncrDetectionStage.__init__(self, **kwargs)
