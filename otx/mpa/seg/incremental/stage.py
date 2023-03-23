# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
from mmcv import ConfigDict

from otx.algorithms.common.adapters.mmcv.utils.config_utils import (
    update_or_add_custom_hook,
)
from otx.algorithms.common.utils.logger import get_logger
from otx.mpa.seg.stage import SegStage

logger = get_logger()


class IncrSegStage(SegStage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def configure_task(self, cfg, training, **kwargs):
        """Adjust settings for task adaptation"""
        super().configure_task(cfg, training, **kwargs)

        new_classes = np.setdiff1d(self.model_classes, self.org_model_classes).tolist()

        # FIXME : can be naive supervised learning (from-scratch ver.)
        # Check if new classes are added
        has_new_class = True if len(new_classes) > 0 else False
        if has_new_class is False:
            ValueError("Incremental learning should have at least one new class!")

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
