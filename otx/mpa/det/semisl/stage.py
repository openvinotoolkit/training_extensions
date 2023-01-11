# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv import ConfigDict

from otx.mpa.det.incremental import IncrDetectionStage
from otx.mpa.utils.config_utils import update_or_add_custom_hook
from otx.mpa.utils.logger import get_logger

logger = get_logger()


class SemiSLDetectionStage(IncrDetectionStage):
    """Patch config to support semi supervised learning for object detection"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def configure_data(self, cfg, data_cfg, training, **kwargs):
        """Patch cfg.data."""
        super().configure_data(cfg, data_cfg, training, **kwargs)
        if training:
            if "unlabeled" in cfg.data:
                if len(cfg.data.unlabeled.get("pipeline", [])) == 0:
                    cfg.data.unlabeled.pipeline = cfg.data.train.pipeline.copy()
                self.configure_unlabeled_dataloader(cfg, self.distributed)

    def configure_task(self, cfg, training, **kwargs):
        logger.info(f"Semi-SL task config!!!!: training={training}")
        super().configure_task(cfg, training, **kwargs)

    def configure_task_cls_incr(self, cfg, task_adapt_type, org_model_classes, model_classes):
        """Patch for class incremental learning.
        Semi supervised learning should support incrmental learning
        """
        if task_adapt_type == "mpa":
            self.configure_bbox_head(cfg, org_model_classes, model_classes)
            self.configure_task_adapt_hook(cfg, org_model_classes, model_classes)
        else:
            src_data_cfg = self.get_data_cfg(cfg, "train")
            src_data_cfg.pop("old_new_indices", None)

    @staticmethod
    def configure_task_adapt_hook(cfg, org_model_classes, model_classes):
        """Add TaskAdaptHook for sampler.

        TaskAdaptHook does not support ComposedDL
        """
        sampler_flag = False
        update_or_add_custom_hook(
            cfg,
            ConfigDict(
                type="TaskAdaptHook",
                src_classes=org_model_classes,
                dst_classes=model_classes,
                model_type=cfg.model.type,
                sampler_flag=sampler_flag,
                efficient_mode=cfg["task_adapt"].get("efficient_mode", False),
            ),
        )
