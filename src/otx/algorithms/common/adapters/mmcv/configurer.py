"""Implementation of class for default patches of mmcv configs."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
from typing import Any, Dict, List

import numpy as np
import torch
from mmcv.runner import CheckpointLoader
from mmcv.utils import Config, ConfigDict
from torch import distributed as dist

from otx.algorithms.common.adapters.mmcv.utils import (
    patch_adaptive_interval_training,
    patch_early_stopping,
    patch_persistent_workers,
)
from otx.algorithms.common.adapters.mmcv.utils.config_utils import (
    recursively_update_cfg,
)
from otx.algorithms.common.utils import append_dist_rank_suffix
from otx.algorithms.common.utils.logger import get_logger

logger = get_logger()


class BaseConfigurer:
    """Base configurer class for mmcv configs."""

    def __init__(self, task, training):
        self.task_adapt_type = None
        self.task_adapt_op = "REPLACE"
        self.org_model_classes = []
        self.model_classes = []
        self.data_classes = []
        self.task = task
        self.training = training
        self.ema_hooks = ["EMAHook", "CustomModelEMAHook"]  # EMA hooks supporting resume

    def configure_base(self, cfg, data_cfg, data_classes, model_classes, **kwargs):
        """Basic configuration work for recipe.

        Patchings in this function are handled task level previously
        This function might need to be re-orgianized
        """

        self.configure_compatibility(cfg, **kwargs)
        patch_adaptive_interval_training(cfg)
        patch_early_stopping(cfg)
        patch_persistent_workers(cfg)

        # update model config -> model label schema
        self.model_classes = model_classes
        self.data_classes = data_classes
        if data_classes is not None:
            train_data_cfg = self.get_data_cfg(data_cfg, "train")
            train_data_cfg["data_classes"] = data_classes
            new_classes = np.setdiff1d(data_classes, model_classes).tolist()
            train_data_cfg["new_classes"] = new_classes

    def configure_compatibilty(self, cfg, **kwargs):
        """Patch for compatibilty with mmX config."""
        raise NotImplementedError

    def configure_device(self, cfg):
        """Setting device for training and inference."""
        cfg.distributed = False
        if torch.distributed.is_initialized():
            cfg.gpu_ids = [int(os.environ["LOCAL_RANK"])]
            if self.training:  # TODO multi GPU is available only in training. Evaluation needs to be supported later.
                cfg.distributed = True
                self.configure_distributed(cfg)
        elif "gpu_ids" not in cfg:
            cfg.gpu_ids = range(1)

        # consider "cuda" and "cpu" device only
        if not torch.cuda.is_available():
            cfg.device = "cpu"
            cfg.gpu_ids = range(-1, 0)
        else:
            cfg.device = "cuda"

    @staticmethod
    def configure_distributed(cfg: Config) -> None:
        """Patching for distributed training."""
        if hasattr(cfg, "dist_params") and cfg.dist_params.get("linear_scale_lr", False):
            new_lr = dist.get_world_size() * cfg.optimizer.lr
            logger.info(
                f"enabled linear scaling rule to the learning rate. \
                changed LR from {cfg.optimizer.lr} to {new_lr}"
            )
            cfg.optimizer.lr = new_lr

    def configure_ckpt(self, cfg, model_ckpt):
        """Patch checkpoint path for pretrained weight.

        Replace cfg.load_from to model_ckpt
        Replace cfg.load_from to pretrained
        Replace cfg.resume_from to cfg.load_from
        """
        if model_ckpt:
            cfg.load_from = self.get_model_ckpt(model_ckpt)
        if cfg.get("resume", False):
            cfg.resume_from = cfg.load_from
            for hook in cfg.custom_hooks:
                if hook.type in self.ema_hooks:
                    hook.resume_from = cfg.resume_from
        if cfg.get("load_from", None) and cfg.model.backbone.get("pretrained", None):
            cfg.model.backbone.pretrained = None

    @staticmethod
    def get_model_ckpt(ckpt_path, new_path=None):
        """Get pytorch model weights."""
        ckpt = CheckpointLoader.load_checkpoint(ckpt_path, map_location="cpu")
        if "model" in ckpt:
            ckpt = ckpt["model"]
            if not new_path:
                new_path = ckpt_path[:-3] + "converted.pth"
            new_path = append_dist_rank_suffix(new_path)
            torch.save(ckpt, new_path)
            return new_path
        return ckpt_path

    def configure_model(self, cfg, ir_options):
        """Patch config's model.

        Change model type to super type
        Patch for OMZ backbones
        """

        if ir_options is None:
            ir_options = {"ir_model_path": None, "ir_weight_path": None, "ir_weight_init": False}

        cfg.model_task = cfg.model.pop("task", self.task)
        if cfg.model_task != self.task:
            raise ValueError(f"Given cfg ({cfg.filename}) is not supported by {self.task} recipe")

        super_type = cfg.model.pop("super_type", None)
        if super_type:
            cfg.model.arch_type = cfg.model.type
            cfg.model.type = super_type

        # OV-plugin
        ir_model_path = ir_options.get("ir_model_path")
        if ir_model_path:

            def is_mmov_model(key, value):
                if key == "type" and value.startswith("MMOV"):
                    return True
                return False

            ir_weight_path = ir_options.get("ir_weight_path", None)
            ir_weight_init = ir_options.get("ir_weight_init", False)
            recursively_update_cfg(
                cfg,
                is_mmov_model,
                {"model_path": ir_model_path, "weight_path": ir_weight_path, "init_weight": ir_weight_init},
            )

    def configure_data(self, cfg, data_cfg):  # noqa: C901
        """Patch cfg.data.

        Merge cfg and data_cfg
        """

        logger.info("configure_data()")
        if data_cfg:
            for subset in data_cfg.data:
                if subset in cfg.data:
                    src_data_cfg = self.get_data_cfg(cfg, subset)
                    new_data_cfg = self.get_data_cfg(data_cfg, subset)
                    for key in new_data_cfg:
                        src_data_cfg[key] = new_data_cfg[key]
                else:
                    raise ValueError(f"{subset} of data_cfg is not in cfg")

    def configure_task(self, cfg):
        """Patch config to support training algorithm."""
        if "task_adapt" in cfg:
            logger.info(f"task config!!!!: training={self.training}")
            self.task_adapt_type = cfg["task_adapt"].get("type", None)
            self.task_adapt_op = cfg["task_adapt"].get("op", "REPLACE")
            self.configure_classes(cfg)

    def configure_classes(self, cfg):
        """Patch classes for model and dataset."""
        org_model_classes = self.get_model_classes(cfg)
        data_classes = self.get_data_classes(cfg)

        # Model classes
        if self.task_adapt_op == "REPLACE":
            if len(data_classes) == 0:
                model_classes = org_model_classes.copy()
            else:
                model_classes = data_classes.copy()
        elif self.task_adapt_op == "MERGE":
            model_classes = org_model_classes + [cls for cls in data_classes if cls not in org_model_classes]
        else:
            raise KeyError(f"{self.task_adapt_op} is not supported for task_adapt options!")

        cfg.task_adapt.final = model_classes
        cfg.model.task_adapt = ConfigDict(
            src_classes=org_model_classes,
            dst_classes=model_classes,
        )

        self.org_model_classes = org_model_classes
        self.model_classes = model_classes
        self.data_classes = data_classes

        self._configure_head(cfg)

    def _configure_head(self, cfg):
        raise NotImplementedError

    def configure_samples_per_gpu(
        self,
        cfg: Config,
        subsets: List[str] = ["train", "val", "test", "unlabeled"],
    ):
        """Settings samples_per_gpu for each dataloader.

        samples_per_gpu can be changed if it is larger than length of datset
        """

        for subset in subsets:
            if cfg.data.get(subset, None):
                dataloader_cfg = cfg.data.get(f"{subset}_dataloader", ConfigDict())
                samples_per_gpu = dataloader_cfg.get("samples_per_gpu", cfg.data.get("samples_per_gpu", 1))

                data_cfg = self.get_data_cfg(cfg, subset)
                if data_cfg.get("otx_dataset") is not None:
                    dataset_len = len(data_cfg.otx_dataset)

                    if getattr(cfg, "distributed", False):
                        dataset_len = dataset_len // dist.get_world_size()

                    # set batch size as a total dataset
                    # if it is smaller than total dataset
                    if dataset_len < samples_per_gpu:
                        dataloader_cfg.samples_per_gpu = dataset_len
                        logger.info(f"{subset}'s samples_per_gpu: {samples_per_gpu} --> {dataset_len}")

                    # drop the last batch if the last batch size is 1
                    # batch size of 1 is a runtime error for training batch normalization layer
                    if subset in ("train", "unlabeled") and dataset_len % samples_per_gpu == 1:
                        dataloader_cfg["drop_last"] = True

                    cfg.data[f"{subset}_dataloader"] = dataloader_cfg

    @staticmethod
    def configure_fp16(cfg: Config):
        """Configure Fp16OptimizerHook and Fp16SAMOptimizerHook."""

        fp16_config = cfg.pop("fp16", None)

        if fp16_config is not None:
            if torch.cuda.is_available():
                optim_type = cfg.optimizer_config.get("type", "OptimizerHook")
                opts: Dict[str, Any] = dict(
                    distributed=getattr(cfg, "distributed", False),
                    **fp16_config,
                )
                if optim_type == "SAMOptimizerHook":
                    opts["type"] = "Fp16SAMOptimizerHook"
                elif optim_type == "OptimizerHook":
                    opts["type"] = "Fp16OptimizerHook"
                else:
                    # does not support optimizerhook type
                    # let mm library handle it
                    cfg.fp16 = fp16_config
                    opts = dict()
                cfg.optimizer_config.update(opts)
            else:
                logger.info("Revert FP16 to FP32 on CPU device")
                if isinstance(cfg, Config):
                    del cfg._cfg_dict["fp16"]  # pylint: disable=protected-access
                elif isinstance(cfg, ConfigDict):
                    del cfg["fp16"]

    @staticmethod
    def configure_compat_cfg(cfg: Config):
        """Modify config to keep the compatibility."""

        global_dataloader_cfg: Dict[str, str] = {}
        global_dataloader_cfg.update(
            {
                k: cfg.data.pop(k)
                for k in list(cfg.data.keys())
                if k
                not in [
                    "train",
                    "val",
                    "test",
                    "unlabeled",
                    "train_dataloader",
                    "val_dataloader",
                    "test_dataloader",
                    "unlabeled_dataloader",
                ]
            }
        )

        for subset in ["train", "val", "test", "unlabeled"]:
            if subset not in cfg.data:
                continue
            dataloader_cfg = cfg.data.get(f"{subset}_dataloader", None)
            if dataloader_cfg is None:
                raise AttributeError(f"{subset}_dataloader is not found in config.")
            dataloader_cfg = Config(cfg_dict={**global_dataloader_cfg, **dataloader_cfg})
            cfg.data[f"{subset}_dataloader"] = dataloader_cfg

    def get_model_classes(self, cfg):
        """Extract trained classes info from checkpoint file.

        MMCV-based models would save class info in ckpt['meta']['CLASSES']
        For other cases, try to get the info from cfg.model.classes (with pop())
        - Which means that model classes should be specified in model-cfg for
          non-MMCV models (e.g. OMZ models)
        """

        def get_model_meta(cfg):
            ckpt_path = cfg.get("load_from", None)
            meta = {}
            if ckpt_path:
                ckpt = CheckpointLoader.load_checkpoint(ckpt_path, map_location="cpu")
                meta = ckpt.get("meta", {})
            return meta

        classes = []
        meta = get_model_meta(cfg)
        # for MPA classification legacy compatibility
        classes = meta.get("CLASSES", [])
        classes = meta.get("classes", classes)
        if classes is None:
            classes = []

        if len(classes) == 0:
            ckpt_path = cfg.get("load_from", None)
            if ckpt_path:
                classes = self.model_classes
        if len(classes) == 0:
            classes = cfg.model.pop("classes", cfg.pop("model_classes", []))
        return classes

    def get_data_classes(self, cfg):
        """Get data classes from train cfg."""
        data_classes = []
        train_cfg = self.get_data_cfg(cfg, "train")
        if "data_classes" in train_cfg:
            data_classes = list(train_cfg.pop("data_classes", []))
        elif "classes" in train_cfg:
            data_classes = list(train_cfg.classes)
        return data_classes

    @staticmethod
    def get_data_cfg(cfg, subset):
        """Get subset's data cfg."""
        assert subset in ["train", "val", "test", "unlabeled"], f"Unknown subset:{subset}"
        if "dataset" in cfg.data[subset]:  # Concat|RepeatDataset
            dataset = cfg.data[subset].dataset
            while hasattr(dataset, "dataset"):
                dataset = dataset.dataset
            return dataset
        return cfg.data[subset]
