# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv import ConfigDict

from otx.mpa.cls.stage import ClsStage, Stage
from otx.mpa.utils.config_utils import update_or_add_custom_hook
from otx.mpa.utils.logger import get_logger

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

        self.adapt_type = cfg["task_adapt"].get("op", "REPLACE")
        train_data_cfg = Stage.get_data_cfg(cfg, "train")
        if training:
            if train_data_cfg.type not in CLASS_INC_DATASET:
                logger.warning(f"Class Incremental Learning for {train_data_cfg.type} is not yet supported!")
            if "new_classes" not in train_data_cfg:
                logger.warning('"new_classes" should be defined for incremental learning w/ current model.')

            if cfg.model.type in WEIGHT_MIX_CLASSIFIER:
                cfg.model.task_adapt = ConfigDict(
                    src_classes=self.model_classes,
                    dst_classes=self.data_classes,
                )
            else:
                logger.warning(f"Weight mixing for {cfg.model.type} is not yet supported!")

            # refine self.dst_class following adapt_type (REPLACE, MERGE)
            self.refine_classes(train_data_cfg)
            cfg.model.head.num_classes = len(self.dst_classes)

            # configure loss, sampler, task_adapt_hook
            self.configure_task_modules(cfg)

        else:  # if eval phase (eval)
            if train_data_cfg.get("new_classes"):
                self.refine_classes(train_data_cfg)
                cfg.model.head.num_classes = len(self.dst_classes)

    def configure_task_modules(self, cfg):
        if not cfg.model.get("multilabel", False) and not cfg.model.get("hierarchical", False):
            efficient_mode = cfg["task_adapt"].get("efficient_mode", True)
            sampler_type = "balanced"
            self.configure_loss(cfg)
        else:
            efficient_mode = cfg["task_adapt"].get("efficient_mode", False)
            sampler_type = "cls_incr"

        if len(set(self.model_classes) & set(self.dst_classes)) == 0 or set(self.model_classes) == set(
            self.dst_classes
        ):
            sampler_flag = False
        else:
            sampler_flag = True

        # Update Task Adapt Hook
        task_adapt_hook = ConfigDict(
            type="TaskAdaptHook",
            src_classes=self.old_classes,
            dst_classes=self.dst_classes,
            model_type=cfg.model.type,
            sampler_flag=sampler_flag,
            sampler_type=sampler_type,
            efficient_mode=efficient_mode,
        )
        update_or_add_custom_hook(cfg, task_adapt_hook)

    def configure_loss(self, cfg):
        if len(set(self.model_classes) & set(self.dst_classes)) == 0 or set(self.model_classes) == set(
            self.dst_classes
        ):
            cfg.model.head.loss = dict(type="CrossEntropyLoss", loss_weight=1.0)
        else:
            cfg.model.head.loss = ConfigDict(
                type="IBLoss",
                num_classes=cfg.model.head.num_classes,
            )
            ib_loss_hook = ConfigDict(
                type="IBLossHook",
                dst_classes=self.dst_classes,
            )
            update_or_add_custom_hook(cfg, ib_loss_hook)

    def refine_classes(self, train_cfg):
        # Get 'new_classes' in data.train_cfg & get 'old_classes' pretreained model meta data CLASSES
        new_classes = train_cfg["new_classes"]
        self.old_classes = self.model_meta["CLASSES"]
        if self.adapt_type == "REPLACE":
            # if 'REPLACE' operation, then self.dst_classes -> data_classes
            self.dst_classes = self.data_classes.copy()
        elif self.adapt_type == "MERGE":
            # if 'MERGE' operation, then self.dst_classes -> old_classes + new_classes (merge)
            self.dst_classes = self.old_classes + [cls for cls in new_classes if cls not in self.old_classes]
        else:
            raise KeyError(f"{self.adapt_type} is not supported for task_adapt options!")
        train_cfg.classes = self.dst_classes
