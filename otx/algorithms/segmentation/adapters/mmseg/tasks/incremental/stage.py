"""Stage for Incremental learning OTX segmentation with MMSEG."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
from mmcv import ConfigDict

from otx.algorithms.common.adapters.mmcv.utils.config_utils import (
    update_or_add_custom_hook,
)
from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.segmentation.adapters.mmseg.tasks.stage import SegStage

logger = get_logger()


class IncrSegStage(SegStage):
    """Calss for incremental learning for segmentation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def configure_task(self, cfg, training):
        """Adjust settings for task adaptation."""
        super().configure_task(cfg, training)

        new_classes = np.setdiff1d(self.model_classes, self.org_model_classes).tolist()

        # Check if new classes are added
        has_new_class = len(new_classes) > 0

        # Update TaskAdaptHook (use incremental sampler)
        task_adapt_hook = ConfigDict(
            type="TaskAdaptHook",
            src_classes=self.org_model_classes,
            dst_classes=self.model_classes,
            model_type=cfg.model.type,
            sampler_flag=has_new_class,
            efficient_mode=cfg["task_adapt"].get("efficient_mode", False),
        )
        update_or_add_custom_hook(cfg, task_adapt_hook)
