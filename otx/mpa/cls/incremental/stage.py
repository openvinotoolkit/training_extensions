# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv import ConfigDict

from otx.algorithms.common.adapters.mmcv.utils.config_utils import (
    update_or_add_custom_hook,
)
from otx.algorithms.common.utils.logger import get_logger
from otx.mpa.cls.stage import ClsStage

logger = get_logger()

CLASS_INC_DATASET = [
    "OTXClsDataset",
    "OTXMultilabelClsDataset",
    "MPAHierarchicalClsDataset",
    "ClsTVDataset",
]
PSEUDO_LABEL_ENABLE_DATASET = ["ClassIncDataset", "ClsTVDataset"]
WEIGHT_MIX_CLASSIFIER = ["SAMImageClassifier"]


class IncrClsStage(ClsStage):
    """Patch config to support incremental learning for object Cls"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def configure_task(self, cfg, training, **kwargs):
        """Patch config to support incremental learning"""
        super().configure_task(cfg, training, **kwargs)
        if "task_adapt" in cfg:
            self.configure_task_adapt(cfg, training, **kwargs)

    # noqa: C901
    def configure_task_adapt(self, cfg, training, **kwargs):
        """Configure for Task Adaptation Task"""
        train_data_cfg = self.get_data_cfg(cfg, "train")
        if training:
            if train_data_cfg.type not in CLASS_INC_DATASET:
                logger.warning(f"Class Incremental Learning for {train_data_cfg.type} is not yet supported!")
            if "new_classes" not in train_data_cfg:
                logger.warning('"new_classes" should be defined for incremental learning w/ current model.')

            if cfg.model.type in WEIGHT_MIX_CLASSIFIER:
                cfg.model.task_adapt = ConfigDict(
                    src_classes=self.org_model_classes,
                    dst_classes=self.model_classes,
                )
            else:
                logger.warning(f"Weight mixing for {cfg.model.type} is not yet supported!")

            train_data_cfg.classes = self.model_classes

            # configure loss, sampler, task_adapt_hook
            self.configure_task_modules(cfg)

    def configure_task_modules(self, cfg):
        if not cfg.model.get("multilabel", False) and not cfg.model.get("hierarchical", False):
            efficient_mode = cfg["task_adapt"].get("efficient_mode", True)
            sampler_type = "balanced"
            self.configure_loss(cfg)
        else:
            efficient_mode = cfg["task_adapt"].get("efficient_mode", False)
            sampler_type = "cls_incr"

        if len(set(self.org_model_classes) & set(self.model_classes)) == 0 or set(self.org_model_classes) == set(
            self.model_classes
        ):
            sampler_flag = False
        else:
            sampler_flag = True

        # Update Task Adapt Hook
        task_adapt_hook = ConfigDict(
            type="TaskAdaptHook",
            src_classes=self.org_model_classes,
            dst_classes=self.model_classes,
            model_type=cfg.model.type,
            sampler_flag=sampler_flag,
            sampler_type=sampler_type,
            efficient_mode=efficient_mode,
        )
        update_or_add_custom_hook(cfg, task_adapt_hook)

    def configure_loss(self, cfg):
        if len(set(self.org_model_classes) & set(self.model_classes)) == 0 or set(self.org_model_classes) == set(
            self.model_classes
        ):
            cfg.model.head.loss = dict(type="CrossEntropyLoss", loss_weight=1.0)
        else:
            cfg.model.head.loss = ConfigDict(
                type="IBLoss",
                num_classes=cfg.model.head.num_classes,
            )
            ib_loss_hook = ConfigDict(
                type="IBLossHook",
                dst_classes=self.model_classes,
            )
            update_or_add_custom_hook(cfg, ib_loss_hook)
