"""Class Incremental Learning configuration mixin."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import List

from mmcv.utils import ConfigDict

from otx.algorithms.common.adapters.mmcv.utils.config_utils import update_or_add_custom_hook


class IncrConfigurerMixin:
    """Patch config to support incremental learning for object detection."""

    org_model_classes: List = []
    model_classes: List = []

    def configure_task(self, cfg, **kwargs):
        """Patch config to support incremental learning."""
        super().configure_task(cfg, **kwargs)
        if "task_adapt" in cfg and self.task_adapt_type == "default_task_adapt":
            self.configure_task_adapt_hook(cfg)

    def configure_task_adapt_hook(self, cfg):
        """Add TaskAdaptHook for sampler."""
        sampler_flag = self.is_incremental()

        update_or_add_custom_hook(
            cfg,
            ConfigDict(
                type="TaskAdaptHook",
                src_classes=self.org_model_classes,
                dst_classes=self.model_classes,
                model_type=cfg.model.type,
                sampler_flag=sampler_flag,
                sampler_type=self.get_sampler_type(cfg),
                efficient_mode=cfg["task_adapt"].get("efficient_mode", False),
                priority="NORMAL",
            ),
        )

    def is_incremental(self) -> bool:
        """Return whether current model classes is increased from original model classes."""
        return len(set(self.org_model_classes) & set(self.model_classes)) > 0 and set(self.org_model_classes) != set(
            self.model_classes
        )

    def get_sampler_type(self, cfg) -> str:
        """Return sampler type."""
        return "cls_incr"
