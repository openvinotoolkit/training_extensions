# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import copy

import numpy as np
import torch
from mmcv import ConfigDict, build_from_cfg
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from otx.mpa.stage import Stage
from otx.mpa.utils.config_utils import recursively_update_cfg, update_or_add_custom_hook
from otx.mpa.utils.data_cpu import MMDataCPU
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


class ClsStage(Stage):
    def configure(self, model_cfg, model_ckpt, data_cfg, training=True, **kwargs):
        """Create MMCV-consumable config from given inputs"""
        logger.info(f"configure: training={training}")

        # Recipe + model
        cfg = self.cfg
        if model_cfg:
            if hasattr(cfg, "model"):
                cfg.merge_from_dict(model_cfg._cfg_dict)
            else:
                cfg.model = copy.deepcopy(model_cfg.model)

        cfg.model_task = cfg.model.pop("task", "classification")
        if cfg.model_task != "classification":
            raise ValueError(f"Given model_cfg ({model_cfg.filename}) is not supported by classification recipe")

        # Checkpoint
        if model_ckpt:
            cfg.load_from = self.get_model_ckpt(model_ckpt)

        if cfg.get("resume", False):
            cfg.resume_from = cfg.load_from

        # OV-plugin
        ir_model_path = kwargs.get("ir_model_path")
        if ir_model_path:

            def is_mmov_model(k, v):
                if k == "type" and v.startswith("MMOV"):
                    return True
                return False

            ir_weight_path = kwargs.get("ir_weight_path", None)
            ir_weight_init = kwargs.get("ir_weight_init", False)
            recursively_update_cfg(
                cfg,
                is_mmov_model,
                {"model_path": ir_model_path, "weight_path": ir_weight_path, "init_weight": ir_weight_init},
            )

        self.configure_model(cfg, training, **kwargs)

        pretrained = kwargs.get("pretrained", None)
        if pretrained and isinstance(pretrained, str):
            logger.info(f"Overriding cfg.load_from -> {pretrained}")
            cfg.load_from = pretrained

        # Data
        if data_cfg:
            cfg.merge_from_dict(data_cfg)
        self.configure_data(cfg, training, **kwargs)

        # Task
        if "task_adapt" in cfg:
            model_meta = self.get_model_meta(cfg)
            model_tasks, dst_classes = self.configure_task(cfg, training, model_meta, **kwargs)
            if model_tasks is not None:
                self.model_tasks = model_tasks
            if dst_classes is not None:
                self.model_classes = dst_classes
        else:
            if "num_classes" not in cfg.data:
                cfg.data.num_classes = len(cfg.data.train.get("classes", []))
            cfg.model.head.num_classes = cfg.data.num_classes

        if cfg.model.head.get("topk", False) and isinstance(cfg.model.head.topk, tuple):
            cfg.model.head.topk = (1,) if cfg.model.head.num_classes < 5 else (1, 5)
            if cfg.model.get("multilabel", False) or cfg.model.get("hierarchical", False):
                cfg.model.head.pop("topk", None)

        # Other hyper-parameters
        if cfg.get("hyperparams", False):
            self.configure_hyperparams(cfg, training, **kwargs)

        return cfg

    @staticmethod
    def configure_model(cfg, training, **kwargs):
        # verify and update model configurations
        # check whether in/out of the model layers require updating

        if cfg.get("load_from", None) and cfg.model.backbone.get("pretrained", None):
            cfg.model.backbone.pretrained = None

        update_required = False
        if cfg.model.get("neck") is not None:
            if cfg.model.neck.get("in_channels") is not None and cfg.model.neck.in_channels <= 0:
                update_required = True
        if not update_required and cfg.model.get("head") is not None:
            if cfg.model.head.get("in_channels") is not None and cfg.model.head.in_channels <= 0:
                update_required = True
        if not update_required:
            return

        # update model layer's in/out configuration
        from mmcv.cnn import MODELS as backbone_reg

        layer = build_from_cfg(cfg.model.backbone, backbone_reg)
        layer.eval()
        input_shape = [3, 224, 224]
        # MMOV model
        if hasattr(layer, "input_shapes"):
            input_shape = next(iter(getattr(layer, "input_shapes").values()))
            input_shape = input_shape[1:]
            if any(i < 0 for i in input_shape):
                input_shape = [3, 244, 244]
        logger.debug(f"input shape for backbone {input_shape}")
        output = layer(torch.rand([1] + list(input_shape)))
        if isinstance(output, (tuple, list)):
            output = output[-1]
        in_channels = output.shape[1]
        if cfg.model.get("neck") is not None:
            if cfg.model.neck.get("in_channels") is not None:
                logger.info(
                    f"'in_channels' config in model.neck is updated from "
                    f"{cfg.model.neck.in_channels} to {in_channels}"
                )
                cfg.model.neck.in_channels = in_channels
                logger.debug(f"input shape for neck {input_shape}")
                from mmcls.models.builder import NECKS as neck_reg

                layer = build_from_cfg(cfg.model.neck, neck_reg)
                layer.eval()
                output = layer(torch.rand(output.shape))
                if isinstance(output, (tuple, list)):
                    output = output[-1]
                in_channels = output.shape[1]
        if cfg.model.get("head") is not None:
            if cfg.model.head.get("in_channels") is not None:
                logger.info(
                    f"'in_channels' config in model.head is updated from "
                    f"{cfg.model.head.in_channels} to {in_channels}"
                )
                cfg.model.head.in_channels = in_channels

            # checking task incremental model configurations

    @staticmethod
    def configure_task(cfg, training, model_meta=None, **kwargs):
        """Configure for Task Adaptation Task"""
        task_adapt_type = cfg["task_adapt"].get("type", None)
        adapt_type = cfg["task_adapt"].get("op", "REPLACE")

        model_tasks, dst_classes = None, None
        model_classes, data_classes = [], []
        train_data_cfg = Stage.get_train_data_cfg(cfg)
        if isinstance(train_data_cfg, list):
            train_data_cfg = train_data_cfg[0]

        model_classes = Stage.get_model_classes(cfg)
        data_classes = Stage.get_data_classes(cfg)
        if model_classes:
            cfg.model.head.num_classes = len(model_classes)
        elif data_classes:
            cfg.model.head.num_classes = len(data_classes)
        model_meta["CLASSES"] = model_classes

        if not train_data_cfg.get("new_classes", False):  # when train_data_cfg doesn't have 'new_classes' key
            new_classes = np.setdiff1d(data_classes, model_classes).tolist()
            train_data_cfg["new_classes"] = new_classes

        if training:
            # if Trainer to Stage configure, training = True
            if train_data_cfg.get("tasks"):
                # Task Adaptation
                if model_meta.get("tasks", False):
                    model_tasks, old_tasks = refine_tasks(train_data_cfg, model_meta, adapt_type)
                else:
                    raise KeyError(f"can not find task meta data from {cfg.load_from}.")
                cfg.model.head.update({"old_tasks": old_tasks})
                # update model.head.tasks with training dataset's tasks if it's configured as None
                if cfg.model.head.get("tasks") is None:
                    logger.info(
                        "'tasks' in model.head is None. updated with configuration on train data "
                        f"{train_data_cfg.get('tasks')}"
                    )
                    cfg.model.head.update({"tasks": train_data_cfg.get("tasks")})
            elif "new_classes" in train_data_cfg:
                # Class-Incremental
                dst_classes, old_classes = refine_cls(train_data_cfg, data_classes, model_meta, adapt_type)
            else:
                raise KeyError('"new_classes" or "tasks" should be defined for incremental learning w/ current model.')

            if task_adapt_type == "mpa":
                if train_data_cfg.type not in CLASS_INC_DATASET:  # task incremental is not supported yet
                    raise NotImplementedError(
                        f"Class Incremental Learning for {train_data_cfg.type} is not yet supported!"
                    )

                if cfg.model.type in WEIGHT_MIX_CLASSIFIER:
                    cfg.model.task_adapt = ConfigDict(
                        src_classes=model_classes,
                        dst_classes=data_classes,
                    )

                # Train dataset config update
                train_data_cfg.classes = dst_classes

                # model configuration update
                cfg.model.head.num_classes = len(dst_classes)

                if not cfg.model.get("multilabel", False) and not cfg.model.get("hierarchical", False):
                    efficient_mode = cfg["task_adapt"].get("efficient_mode", True)
                    sampler_type = "balanced"

                    if len(set(model_classes) & set(dst_classes)) == 0 or set(model_classes) == set(dst_classes):
                        cfg.model.head.loss = dict(type="CrossEntropyLoss", loss_weight=1.0)
                    else:
                        cfg.model.head.loss = ConfigDict(
                            type="IBLoss",
                            num_classes=cfg.model.head.num_classes,
                        )
                        ib_loss_hook = ConfigDict(
                            type="IBLossHook",
                            dst_classes=dst_classes,
                        )
                        update_or_add_custom_hook(cfg, ib_loss_hook)
                else:
                    efficient_mode = cfg["task_adapt"].get("efficient_mode", False)
                    sampler_type = "cls_incr"

                if len(set(model_classes) & set(dst_classes)) == 0 or set(model_classes) == set(dst_classes):
                    sampler_flag = False
                else:
                    sampler_flag = True

                # Update Task Adapt Hook
                task_adapt_hook = ConfigDict(
                    type="TaskAdaptHook",
                    src_classes=old_classes,
                    dst_classes=dst_classes,
                    model_type=cfg.model.type,
                    sampler_flag=sampler_flag,
                    sampler_type=sampler_type,
                    efficient_mode=efficient_mode,
                )
                update_or_add_custom_hook(cfg, task_adapt_hook)

        else:  # if not training phase (eval)
            if train_data_cfg.get("tasks"):
                if model_meta.get("tasks", False):
                    cfg.model.head["tasks"] = model_meta["tasks"]
                else:
                    raise KeyError(f"can not find task meta data from {cfg.load_from}.")
            elif train_data_cfg.get("new_classes"):
                dst_classes, _ = refine_cls(train_data_cfg, data_classes, model_meta, adapt_type)
                cfg.model.head.num_classes = len(dst_classes)

        # Pseudo label augmentation
        pre_stage_res = kwargs.get("pre_stage_res", None)
        if pre_stage_res:
            logger.info(f"pre-stage dataset: {pre_stage_res}")
            if train_data_cfg.type not in PSEUDO_LABEL_ENABLE_DATASET:
                raise NotImplementedError(f"Pseudo label loading for {train_data_cfg.type} is not yet supported!")
            train_data_cfg.pre_stage_res = pre_stage_res
            if train_data_cfg.get("tasks"):
                train_data_cfg.model_tasks = model_tasks
                cfg.model.head.old_tasks = old_tasks
            elif train_data_cfg.get("CLASSES"):
                train_data_cfg.dst_classes = dst_classes
                cfg.data.val.dst_classes = dst_classes
                cfg.data.test.dst_classes = dst_classes
                cfg.model.head.num_classes = len(train_data_cfg.dst_classes)
                cfg.model.head.num_old_classes = len(old_classes)
        return model_tasks, dst_classes

    @staticmethod
    def configure_hyperparams(cfg, training, **kwargs):
        hyperparams = kwargs.get("hyperparams", None)
        if hyperparams is not None:
            bs = hyperparams.get("bs", None)
            if bs is not None:
                cfg.data.samples_per_gpu = bs

            lr = hyperparams.get("lr", None)
            if lr is not None:
                cfg.optimizer.lr = lr

    def _put_model_on_gpu(self, model, cfg):
        if torch.cuda.is_available():
            model = model.cuda()
            if self.distributed:
                # put model on gpus
                find_unused_parameters = cfg.get("find_unused_parameters", False)
                # Sets the `find_unused_parameters` parameter in
                # torch.nn.parallel.DistributedDataParallel
                model = MMDistributedDataParallel(
                    model,
                    device_ids=[torch.cuda.current_device()],
                    broadcast_buffers=False,
                    find_unused_parameters=find_unused_parameters,
                )
            else:
                model = MMDataParallel(model.cuda(), device_ids=[0])
        else:
            model = MMDataCPU(model)

        return model


def refine_tasks(train_cfg, meta, adapt_type):
    new_tasks = train_cfg["tasks"]
    if adapt_type == "REPLACE":
        old_tasks = {}
        model_tasks = new_tasks
    elif adapt_type == "MERGE":
        old_tasks = meta["tasks"]
        model_tasks = copy.deepcopy(old_tasks)
        for task, cls in new_tasks.items():
            if model_tasks.get(task):
                model_tasks[task] = model_tasks[task] + [c for c in cls if c not in model_tasks[task]]
            else:
                model_tasks.update({task: cls})
    else:
        raise KeyError(f"{adapt_type} is not supported for task_adapt options!")
    return model_tasks, old_tasks


def refine_cls(train_cfg, data_classes, meta, adapt_type):
    # Get 'new_classes' in data.train_cfg & get 'old_classes' pretreained model meta data CLASSES
    new_classes = train_cfg["new_classes"]
    old_classes = meta["CLASSES"]
    if adapt_type == "REPLACE":
        # if 'REPLACE' operation, then dst_classes -> data_classes
        dst_classes = data_classes.copy()
    elif adapt_type == "MERGE":
        # if 'MERGE' operation, then dst_classes -> old_classes + new_classes (merge)
        dst_classes = old_classes + [cls for cls in new_classes if cls not in old_classes]
    else:
        raise KeyError(f"{adapt_type} is not supported for task_adapt options!")
    return dst_classes, old_classes
