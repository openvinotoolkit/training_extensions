# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv import ConfigDict

from otx.algorithms.common.adapters.mmcv.utils.config_utils import (
    update_or_add_custom_hook,
)
from otx.algorithms.common.utils.logger import get_logger
from otx.mpa.det.stage import DetectionStage

logger = get_logger()


class IncrDetectionStage(DetectionStage):
    """Patch config to support incremental learning for object detection"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def configure_task(self, cfg, training, **kwargs):
        """Patch config to support incremental learning"""
        super().configure_task(cfg, training, **kwargs)
        if "task_adapt" in cfg and self.task_adapt_type == "mpa":
            self.configure_task_adapt_hook(cfg)

    def configure_task_adapt_hook(self, cfg):
        """Add TaskAdaptHook for sampler."""
        sampler_flag = True
        if len(set(self.org_model_classes) & set(self.model_classes)) == 0 or set(self.org_model_classes) == set(
            self.model_classes
        ):
            sampler_flag = False
        update_or_add_custom_hook(
            cfg,
            ConfigDict(
                type="TaskAdaptHook",
                src_classes=self.org_model_classes,
                dst_classes=self.model_classes,
                model_type=cfg.model.type,
                sampler_flag=sampler_flag,
                efficient_mode=cfg["task_adapt"].get("efficient_mode", False),
            ),
        )
