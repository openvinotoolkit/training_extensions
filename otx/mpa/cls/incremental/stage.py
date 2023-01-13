# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import copy

from mmcv import ConfigDict

from otx.mpa.cls.stage import ClsStage, Stage
from otx.mpa.utils.config_utils import update_or_add_custom_hook
from otx.mpa.utils.logger import get_logger

logger = get_logger()

CLASS_INC_DATASET = [
    "MPAClsDataset",
    "MPAMultilabelClsDataset",
    "MPAHierarchicalClsDataset",
    "ClsDirDataset",
    "ClsTVDataset",
]
PSEUDO_LABEL_ENABLE_DATASET = ["ClassIncDataset", "LwfTaskIncDataset", "ClsTVDataset"]
WEIGHT_MIX_CLASSIFIER = ["SAMImageClassifier"]


class IncrClsStage(ClsStage):
    """Patch config to support incremental learning for object Cls"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def configure_task(self, cfg, training, **kwargs):
        """Patch config to support incremental learning"""
        super().configure_task(cfg, training, **kwargs)
        if "task_adapt" in cfg:
            self.sub_configure_task(cfg, training, **kwargs)

    # noqa: C901
    def sub_configure_task(self, cfg, training, **kwargs):
        """Configure for Task Adaptation Task"""

        self.task_type = cfg["task_adapt"].get("type", None)
        self.adapt_type = cfg["task_adapt"].get("op", "REPLACE")

        model_tasks, self.dst_classes = None, None
        train_data_cfg = Stage.get_data_cfg(cfg, "train")

        if training:
            # if Trainer to Stage configure, training = True
            if train_data_cfg.get("tasks"):
                # Task Adaptation
                if self.model_meta.get("tasks", False):
                    self.refine_tasks(train_data_cfg)
                else:
                    raise KeyError(f"can not find task meta data from {cfg.load_from}.")
                cfg.model.head.update({"old_tasks": self.old_tasks})
                # update model.head.tasks with training dataset's tasks if it's configured as None
                if cfg.model.head.get("tasks") is None:
                    logger.info(
                        "'tasks' in model.head is None. updated with configuration on train data "
                        f"{train_data_cfg.get('tasks')}"
                    )
                    cfg.model.head.update({"tasks": train_data_cfg.get("tasks")})
            elif "new_classes" in train_data_cfg:
                # Class-Incremental
                self.refine_cls(train_data_cfg)
            else:
                raise KeyError('"new_classes" or "tasks" should be defined for incremental learning w/ current model.')

            if self.task_type == "mpa":
                if train_data_cfg.type not in CLASS_INC_DATASET:  # task incremental is not supported yet
                    raise NotImplementedError(
                        f"Class Incremental Learning for {train_data_cfg.type} is not yet supported!"
                    )

                if cfg.model.type in WEIGHT_MIX_CLASSIFIER:
                    cfg.model.task_adapt = ConfigDict(
                        src_classes=self.model_classes,
                        dst_classes=self.data_classes,
                    )

                # Train dataset config update
                train_data_cfg.classes = self.dst_classes

                # model configuration update
                cfg.model.head.num_classes = len(self.dst_classes)
                self.configure_task_adapt(cfg)

        else:  # if not training phase (eval)
            if train_data_cfg.get("tasks"):
                if self.model_meta.get("tasks", False):
                    cfg.model.head["tasks"] = self.model_meta["tasks"]
                else:
                    raise KeyError(f"can not find task meta data from {cfg.load_from}.")
            elif train_data_cfg.get("new_classes"):
                self.refine_cls(train_data_cfg)
                cfg.model.head.num_classes = len(self.dst_classes)

        self.model_tasks = model_tasks
        self.model_classes = self.dst_classes
        self.configure_pseudo_label(cfg, **kwargs)

    def configure_task_adapt(self, cfg):
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

    def configure_pseudo_label(self, cfg, **kwargs):
        # Pseudo label augmentation
        train_data_cfg = Stage.get_data_cfg(cfg, "train")
        pre_stage_res = kwargs.get("pre_stage_res", None)
        if pre_stage_res:
            logger.info(f"pre-stage dataset: {pre_stage_res}")
            if train_data_cfg.type not in PSEUDO_LABEL_ENABLE_DATASET:
                raise NotImplementedError(f"Pseudo label loading for {train_data_cfg.type} is not yet supported!")
            train_data_cfg.pre_stage_res = pre_stage_res
            if train_data_cfg.get("tasks"):
                train_data_cfg.model_tasks = self.model_tasks
                cfg.model.head.old_tasks = self.old_tasks
            elif train_data_cfg.get("CLASSES"):
                train_data_cfg.dst_classes = self.dst_classes
                cfg.data.val.dst_classes = self.dst_classes
                cfg.data.test.dst_classes = self.dst_classes
                cfg.model.head.num_classes = len(self.dst_classes)
                cfg.model.head.num_old_classes = len(self.old_classes)

    def refine_tasks(self, train_cfg):
        new_tasks = train_cfg["tasks"]
        if self.adapt_type == "REPLACE":
            old_tasks = {}
            model_tasks = new_tasks
        elif self.adapt_type == "MERGE":
            old_tasks = self.model_meta["tasks"]
            model_tasks = copy.deepcopy(old_tasks)
            for task, cls in new_tasks.items():
                if model_tasks.get(task):
                    model_tasks[task] = model_tasks[task] + [c for c in cls if c not in model_tasks[task]]
                else:
                    model_tasks.update({task: cls})
        else:
            raise KeyError(f"{self.adapt_type} is not supported for task_adapt options!")
        self.model_tasks = model_tasks
        self.old_tasks = old_tasks

    def refine_cls(self, train_cfg):
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
