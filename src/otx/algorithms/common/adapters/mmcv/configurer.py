"""Implementation of class for default patches of mmcv configs."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from mmcv.runner import CheckpointLoader
from mmcv.utils import Config, ConfigDict
from torch import distributed as dist

from otx.algorithms.common.adapters.mmcv.utils import (
    patch_adaptive_interval_training,
    patch_early_stopping,
    patch_persistent_workers,
    remove_from_configs_by_type,
)
from otx.algorithms.common.adapters.mmcv.utils.config_utils import (
    InputSizeManager,
    patch_color_conversion,
    patch_from_hyperparams,
    recursively_update_cfg,
    update_or_add_custom_hook,
)
from otx.algorithms.common.tasks.base_task import OnHookInitialized
from otx.algorithms.common.utils import (
    UncopiableDefaultDict,
    append_dist_rank_suffix,
    is_hpu_available,
    is_xpu_available,
)
from otx.algorithms.common.utils.data import compute_robust_dataset_statistics
from otx.api.usecases.reporting.time_monitor_callback import TimeMonitorCallback
from otx.core.data import caching
from otx.utils.logger import get_logger

logger = get_logger()


class BaseConfigurer:
    """Base configurer class for mmcv configs."""

    def __init__(
        self,
        task: str,
        training: bool,
        export: bool,
        override_configs: Dict[str, str],
        on_hook_initialized: OnHookInitialized,
        time_monitor: Optional[TimeMonitorCallback],
        learning_curves: UncopiableDefaultDict,
    ):
        self.task_adapt_type: Optional[str] = None
        self.task_adapt_op: str = "REPLACE"
        self.org_model_classes: List[str] = []
        self.model_classes: List[str] = []
        self.data_classes: List[str] = []
        self.task: str = task
        self.training: bool = training
        self.export: bool = export
        self.ema_hooks: List[str] = ["EMAHook", "CustomModelEMAHook"]  # EMA hooks supporting resume
        self.override_configs: Dict[str, str] = override_configs
        self.on_hook_initialized: OnHookInitialized = on_hook_initialized
        self.time_monitor: Optional[TimeMonitorCallback] = time_monitor
        self.learning_curves: UncopiableDefaultDict = learning_curves

    def configure(
        self,
        cfg: Config,
        data_pipeline_path: str,
        hyperparams_from_otx: ConfigDict,
        model_ckpt_path: str,
        data_cfg: Config,
        ir_options: Optional[Config] = None,
        data_classes: Optional[List[str]] = None,
        model_classes: Optional[List[str]] = None,
        input_size: Optional[Tuple[int, int]] = None,
        **kwargs: Dict[Any, Any],
    ) -> Config:
        """Create MMCV-consumable config from given inputs."""
        logger.info(f"configure!: training={self.training}")

        cfg.model_task = cfg.model.pop("task", self.task)
        if cfg.model_task != self.task:
            raise ValueError(f"Given cfg ({cfg.filename}) is not supported by {self.task} recipe")

        self.merge_configs(cfg, data_cfg, data_pipeline_path, hyperparams_from_otx, **kwargs)

        self.configure_ckpt(cfg, model_ckpt_path)
        self.configure_env(cfg)
        self.configure_data_pipeline(cfg, input_size, model_ckpt_path, **kwargs)
        self.configure_recipe(cfg, **kwargs)
        self.configure_model(cfg, data_classes, model_classes, ir_options, **kwargs)
        self.configure_hooks(
            cfg,
        )
        self.configure_compat_cfg(cfg)
        return cfg

    def merge_configs(self, cfg, data_cfg, data_pipeline_path, hyperparams_from_otx, **kwargs):
        """Merge model cfg, data_pipeline cfg, data_cfg, and hyperparams from otx cli."""

        logger.debug("merge_configs()")
        if os.path.isfile(data_pipeline_path):
            data_pipeline_cfg = Config.fromfile(data_pipeline_path)
            cfg.merge_from_dict(data_pipeline_cfg)
        else:
            raise FileNotFoundError(f"data_pipeline: {data_pipeline_path} not founded")

        self.override_from_hyperparams(cfg, hyperparams_from_otx, **kwargs)

        if data_cfg:
            for subset in data_cfg.data:
                if subset in cfg.data:
                    src_data_cfg = self.get_subset_data_cfg(cfg, subset)
                    new_data_cfg = self.get_subset_data_cfg(data_cfg, subset)
                    for key in new_data_cfg:
                        src_data_cfg[key] = new_data_cfg[key]
                else:
                    raise ValueError(f"{subset} of data_cfg is not in cfg")

    def override_from_hyperparams(self, config, hyperparams, **kwargs):
        """Override config using hyperparams from OTX CLI."""
        if not self.export:
            patch_from_hyperparams(config, hyperparams, **kwargs)

    def configure_ckpt(self, cfg, model_ckpt_path):
        """Patch checkpoint path for pretrained weight.

        Replace cfg.load_from to model_ckpt_path
        Replace cfg.load_from to pretrained
        Replace cfg.resume_from to cfg.load_from
        """
        if model_ckpt_path:
            cfg.load_from = self.get_model_ckpt(model_ckpt_path)
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

    def configure_env(self, cfg):
        """Configuration for environment settings."""

        patch_persistent_workers(cfg)
        self.configure_device(cfg)
        self.configure_samples_per_gpu(cfg)

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

        # consider "cuda", "xpu", "hpu" and "cpu" device only
        if is_hpu_available():
            cfg.device = "hpu"
        elif torch.cuda.is_available():
            cfg.device = "cuda"
        elif is_xpu_available():
            cfg.device = "xpu"
        else:
            cfg.device = "cpu"
            cfg.gpu_ids = range(-1, 0)

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

                data_cfg = self.get_subset_data_cfg(cfg, subset)
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

    def configure_data_pipeline(self, cfg, input_size, model_ckpt_path, **kwargs):
        """Configuration data pipeline settings."""

        patch_color_conversion(cfg)
        self.configure_input_size(cfg, input_size, model_ckpt_path, self.training)

    def configure_recipe(self, cfg, **kwargs):
        """Configuration training recipe settings."""

        patch_adaptive_interval_training(cfg)
        patch_early_stopping(cfg)
        self.configure_fp16(cfg)

    @staticmethod
    def configure_fp16(cfg: Config):
        """Configure Fp16OptimizerHook and Fp16SAMOptimizerHook."""

        fp16_config = cfg.pop("fp16", None)
        # workaround to forward FP16 config to mmapi.train funcitons
        cfg.fp16_ = fp16_config

        optim_type = cfg.optimizer_config.get("type", "OptimizerHook")
        distributed = getattr(cfg, "distributed", False)
        opts: Dict[str, Any] = {}
        if fp16_config is not None:
            if is_hpu_available():
                if optim_type == "SAMOptimizerHook":
                    # TODO (sungchul): consider SAM optimizer
                    logger.warning("SAMOptimizerHook is not supported on HPU. Changed to OptimizerHook.")
                opts["type"] = "HPUOptimizerHook"
                cfg.optimizer_config.update(opts)
            elif is_xpu_available():
                if optim_type == "SAMOptimizerHook":
                    logger.warning("SAMOptimizerHook is not supported on XPU yet, changed to OptimizerHook.")
                    opts["type"] = "OptimizerHook"
                cfg.optimizer_config.update(opts)
                logger.warning("XPU doesn't support mixed precision training currently.")
            elif torch.cuda.is_available():
                opts.update({"distributed": distributed, **fp16_config})
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

        elif is_hpu_available():
            if distributed:
                opts["type"] = "HPUDistOptimizerHook"
            else:
                opts["type"] = "HPUOptimizerHook"
            cfg.optimizer_config.update(opts)

        else:
            logger.info("Revert FP16 to FP32 on CPU device")

    def configure_model(self, cfg, data_classes, model_classes, ir_options, **kwargs):
        """Configuration model config settings."""

        self.model_classes = model_classes
        self.data_classes = data_classes
        if data_classes is not None:
            train_data_cfg = self.get_subset_data_cfg(cfg, "train")
            train_data_cfg["data_classes"] = data_classes
            new_classes = np.setdiff1d(data_classes, model_classes).tolist()
            train_data_cfg["new_classes"] = new_classes
        self.configure_backbone(cfg, ir_options)
        self.configure_task(cfg, **kwargs)

    def configure_backbone(self, cfg, ir_options):
        """Patch config's model.

        Change model type to super type
        Patch for OMZ backbones
        """

        if ir_options is None:
            ir_options = {"ir_model_path": None, "ir_weight_path": None, "ir_weight_init": False}

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

    def configure_task(self, cfg, **kwargs):
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

    def configure_hooks(
        self,
        cfg,
    ):
        """Add or update hooks."""

        if "custom_hooks" in self.override_configs:
            override_custom_hooks = self.override_configs.pop("custom_hooks")
            for override_custom_hook in override_custom_hooks:
                update_or_add_custom_hook(cfg, ConfigDict(override_custom_hook))
        if len(self.override_configs) > 0:
            logger.info(f"before override configs merging = {cfg}")
            cfg.merge_from_dict(self.override_configs)
            logger.info(f"after override configs merging = {cfg}")

        # add Cancel training hook
        update_or_add_custom_hook(
            cfg,
            ConfigDict(type="CancelInterfaceHook", init_callback=self.on_hook_initialized),
        )
        if self.time_monitor is not None:
            update_or_add_custom_hook(
                cfg,
                ConfigDict(
                    type="OTXProgressHook",
                    time_monitor=self.time_monitor,
                    verbose=True,
                    priority=71,
                ),
            )
        cfg.log_config.hooks.append({"type": "OTXLoggerHook", "curves": self.learning_curves})
        if hasattr(cfg, "algo_backend"):
            self._update_caching_modules(cfg)

        # Update adaptive repeat
        if not self.training:
            remove_from_configs_by_type(cfg.custom_hooks, "AdaptiveRepeatDataHook")
            return
        for custom_hook in cfg.custom_hooks:
            if custom_hook["type"] == "AdaptiveRepeatDataHook":
                data_cfg = cfg.get("data", {})
                bs = data_cfg.get("train_dataloader", {}).get("samples_per_gpu", None)
                bs = bs if bs is not None else data_cfg.get("samples_per_gpu", 0)
                custom_hook["train_batch_size"] = bs
                custom_hook["train_data_size"] = len(data_cfg.get("train", {}).get("otx_dataset", []))
                break

    @staticmethod
    def _update_caching_modules(cfg: Config) -> None:
        def _find_max_num_workers(cfg: dict):
            num_workers = [0]
            for key, value in cfg.items():
                if key == "workers_per_gpu" and isinstance(value, int):
                    num_workers += [value]
                elif isinstance(value, dict):
                    num_workers += [_find_max_num_workers(value)]

            return max(num_workers)

        def _get_mem_cache_size(cfg):
            if not hasattr(cfg.algo_backend, "mem_cache_size"):
                return 0

            return cfg.algo_backend.mem_cache_size

        max_num_workers = _find_max_num_workers(cfg.data)
        mem_cache_size = _get_mem_cache_size(cfg)

        mode = "multiprocessing" if max_num_workers > 0 else "singleprocessing"
        caching.MemCacheHandlerSingleton.create(mode, mem_cache_size)

        update_or_add_custom_hook(
            cfg,
            ConfigDict(type="MemCacheHook", priority="VERY_LOW"),
        )

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
        # for OTX classification legacy compatibility
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
        train_cfg = self.get_subset_data_cfg(cfg, "train")
        if "data_classes" in train_cfg:
            data_classes = list(train_cfg.pop("data_classes", []))
        elif "classes" in train_cfg:
            data_classes = list(train_cfg.classes)
        return data_classes

    @staticmethod
    def get_subset_data_cfg(cfg, subset):
        """Get subset's data cfg."""
        assert subset in ["train", "val", "test", "unlabeled"], f"Unknown subset:{subset}"
        if "dataset" in cfg.data[subset]:  # Concat|RepeatDataset
            dataset = cfg.data[subset].dataset
            while hasattr(dataset, "dataset"):
                dataset = dataset.dataset
            return dataset
        return cfg.data[subset]

    @staticmethod
    def adapt_input_size_to_dataset(
        cfg, input_size_manager: InputSizeManager, downscale_only: bool = True, use_annotations: bool = False
    ) -> Optional[Tuple[int, int]]:
        """Compute appropriate model input size w.r.t. dataset statistics.

        Args:
            cfg (Dict): Global configuration.
            input_size_manager: (InputSizeManager): Pre-configured input size manager
            downscale_only (bool) : Whether to allow only smaller size than default setting. Defaults to True.
            use_annotations (bool): Whether to consider annotation shapes to compute input size. Defaults to False.

        Returns:
            Tuple[int, int]: (width, height) or None
        """

        data_cfg = BaseConfigurer.get_subset_data_cfg(cfg, "train")
        dataset = data_cfg.get("otx_dataset", None)
        if dataset is None:
            return None

        stat = compute_robust_dataset_statistics(dataset, use_annotations)
        if not stat:
            return None

        def format_float(obj):
            if isinstance(obj, float):
                return f"{obj:.2f}"
            if isinstance(obj, dict):
                return {k: format_float(v) for k, v in obj.items()}
            return obj

        logger.info(f"Dataset stat: {json.dumps(format_float(stat), indent=4)}")

        # Fit to typical large image size (conservative)
        # -> "avg" size might be preferrable for efficiency
        image_size = stat["image"]["robust_max"]
        object_size = None
        if use_annotations and stat["annotation"]:
            # Refine using annotation shape size stat
            # Fit to typical small object size (conservative)
            # -> "avg" size might be preferrable for efficiency
            object_size = stat["annotation"].get("size_of_shape", {}).get("robust_min", None)

        return input_size_manager.adapt_input_size_to_dataset(image_size, object_size, downscale_only)
