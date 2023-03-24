# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import importlib
import json
import os
import os.path as osp
import random
import time
from typing import Any, Callable, Dict, Optional

import mmcv
import numpy as np
import torch
from mmcv import Config, ConfigDict
from mmcv.runner import CheckpointLoader, wrap_fp16_model
from torch import distributed as dist
from torch.utils.data import Dataset

from otx.algorithms.common.adapters.mmcv.utils import (
    build_dataloader,
    build_dataset,
    get_data_cfg,
)
from otx.algorithms.common.adapters.mmcv.utils.config_utils import (
    MPAConfig,
    update_or_add_custom_hook,
)
from otx.algorithms.common.utils.logger import config_logger, get_logger

from .registry import STAGES

logger = get_logger()


def _set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Training seed was set to {seed} w/ deterministic={deterministic}.")
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_available_types():
    types = []
    for k, v in STAGES.module_dict.items():
        # logger.info(f'key [{k}] = value[{v}]')
        types.append(k)
    return types


MODEL_TASK = {"classification": "mmcls", "detection": "mmdet", "segmentation": "mmseg"}

# @STAGES.register_module()
class Stage(object):
    MODEL_BUILDER = None

    def __init__(self, name, mode, config, common_cfg={}, index=0, **kwargs):
        logger.debug(f"init stage with: {name}, {mode}, {config}, {common_cfg}, {index}, {kwargs}")
        # the name of 'config' cannot be changed to such as 'config_file'
        # because it is defined as 'config' in recipe file.....
        self.name = name
        self.mode = mode
        self.index = index
        self.input = kwargs.pop("input", {})  # input_map?? input_dict? just input?
        self.output_keys = kwargs.pop("output", [])
        self._distributed = False

        if common_cfg is None:
            common_cfg = dict(output_path="logs")

        if not isinstance(common_cfg, dict):
            raise TypeError(f"common_cfg should be the type of dict but {type(common_cfg)}")
        else:
            if common_cfg.get("output_path") is None:
                logger.info("output_path is not set in common_cfg. set it to 'logs' as default")
                common_cfg["output_path"] = "logs"

        self.output_prefix = common_cfg["output_path"]
        self.output_suffix = f"stage{self.index:02d}_{self.name}"

        # # Work directory
        # work_dir = os.path.join(self.output_prefix, self.output_suffix)
        # mmcv.mkdir_or_exist(os.path.abspath(work_dir))

        if isinstance(config, Config):
            cfg = config
        elif isinstance(config, dict):
            cfg = Config(cfg_dict=config)
        elif isinstance(config, str):
            if os.path.exists(config):
                cfg = MPAConfig.fromfile(config)
            else:
                err_msg = f"cannot find configuration file {config}"
                logger.error(err_msg)
                raise ValueError(err_msg)
        else:
            err_msg = "'config' argument could be one of the \
                       [dictionary, Config object, or string of the cfg file path]"
            logger.error(err_msg)
            raise ValueError(err_msg)

        cfg.merge_from_dict(common_cfg)

        if len(kwargs) > 0:
            addtional_dict = {}
            logger.info("found override configurations for the stage")
            for k, v in kwargs.items():
                addtional_dict[k] = v
                logger.info(f"\t{k}: {v}")
            cfg.merge_from_dict(addtional_dict)

        max_epochs = -1
        if hasattr(cfg, "total_epochs"):
            max_epochs = cfg.pop("total_epochs")
        if hasattr(cfg, "runner"):
            if hasattr(cfg.runner, "max_epochs"):
                if max_epochs != -1:
                    max_epochs = min(max_epochs, cfg.runner.max_epochs)
                else:
                    max_epochs = cfg.runner.max_epochs
        if max_epochs > 0:
            if cfg.runner.max_epochs != max_epochs:
                cfg.runner.max_epochs = max_epochs
                logger.info(f"The maximum number of epochs is adjusted to {max_epochs}.")
            if hasattr(cfg, "checkpoint_config"):
                if hasattr(cfg.checkpoint_config, "interval"):
                    if cfg.checkpoint_config.interval > max_epochs:
                        logger.warning(
                            f"adjusted checkpoint interval from {cfg.checkpoint_config.interval} to {max_epochs} \
                            since max_epoch is shorter than ckpt interval configuration"
                        )
                        cfg.checkpoint_config.interval = max_epochs

        if cfg.get("seed", None) is not None:
            _set_random_seed(cfg.seed, deterministic=cfg.get("deterministic", False))
        else:
            cfg.seed = None

        # Work directory
        work_dir = cfg.get("work_dir", "")
        work_dir = os.path.join(self.output_prefix, work_dir if work_dir else "", self.output_suffix)
        cfg.work_dir = os.path.abspath(work_dir)
        logger.info(f"work dir = {cfg.work_dir}")
        mmcv.mkdir_or_exist(os.path.abspath(work_dir))

        # config logger replace hook
        hook_cfg = ConfigDict(type="LoggerReplaceHook")
        update_or_add_custom_hook(cfg, hook_cfg)

        self.cfg = cfg

        self.__init_device()

    def __init_device(self):
        if torch.distributed.is_initialized():
            self._distributed = True
            self.cfg.gpu_ids = [int(os.environ["LOCAL_RANK"])]
        elif "gpu_ids" not in self.cfg:
            gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES")
            logger.info(f"CUDA_VISIBLE_DEVICES = {gpu_ids}")
            if gpu_ids is not None:
                self.cfg.gpu_ids = range(len(gpu_ids.split(",")))
            else:
                self.cfg.gpu_ids = range(1)

        # consider "cuda" and "cpu" device only
        if not torch.cuda.is_available():
            self.cfg.device = "cpu"
            self.cfg.gpu_ids = range(-1, 0)
        else:
            self.cfg.device = "cuda"

    @property
    def distributed(self):
        return self._distributed

    def run(self, **kwargs):
        raise NotImplementedError

    def _init_logger(self, **kwargs):
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        config_logger(os.path.join(self.cfg.work_dir, f"{timestamp}.log"), level=self.cfg.log_level)
        logger.info(f"configured logger at {self.cfg.work_dir} with named {timestamp}.log")
        return logger

    @staticmethod
    def configure_data(cfg, training, **kwargs):
        # update data configuration using image options
        def configure_split(target):
            def update_transform(opt, pipeline, idx, transform):
                if isinstance(opt, dict):
                    if "_delete_" in opt.keys() and opt.get("_delete_", False):
                        # if option include _delete_=True, remove this transform from pipeline
                        logger.info(f"configure_data: {transform['type']} is deleted")
                        del pipeline[idx]
                        return
                    logger.info(f"configure_data: {transform['type']} is updated with {opt}")
                    transform.update(**opt)

            def update_config(src, pipeline_options):
                logger.info(f"update_config() {pipeline_options}")
                if src.get("pipeline") is not None or (
                    src.get("dataset") is not None and src.get("dataset").get("pipeline") is not None
                ):
                    if src.get("pipeline") is not None:
                        pipeline = src.get("pipeline", None)
                    else:
                        pipeline = src.get("dataset").get("pipeline")
                    if isinstance(pipeline, list):
                        for idx, transform in enumerate(pipeline):
                            for opt_key, opt in pipeline_options.items():
                                if transform["type"] == opt_key:
                                    update_transform(opt, pipeline, idx, transform)
                    elif isinstance(pipeline, dict):
                        for _, pipe in pipeline.items():
                            for idx, transform in enumerate(pipe):
                                for opt_key, opt in pipeline_options.items():
                                    if transform["type"] == opt_key:
                                        update_transform(opt, pipe, idx, transform)
                    else:
                        raise NotImplementedError(f"pipeline type of {type(pipeline)} is not supported")
                else:
                    logger.info("no pipeline in the data split")

            split = cfg.data.get(target)
            if split is not None:
                if isinstance(split, list):
                    for sub_item in split:
                        update_config(sub_item, pipeline_options)
                elif isinstance(split, dict):
                    update_config(split, pipeline_options)
                else:
                    logger.warning(f"type of split '{target}'' should be list or dict but {type(split)}")

        logger.info("configure_data()")
        logger.debug(f"[args] {cfg.data}")
        pipeline_options = cfg.data.pop("pipeline_options", None)
        if pipeline_options is not None and isinstance(pipeline_options, dict):
            configure_split("train")
            configure_split("val")
            if not training:
                configure_split("test")
            configure_split("unlabeled")

    def configure_ckpt(self, cfg, model_ckpt, pretrained=None):
        """Patch checkpoint path for pretrained weight.
        Replace cfg.load_from to model_ckpt
        Replace cfg.load_from to pretrained
        Replace cfg.resume_from to cfg.load_from
        """
        if model_ckpt:
            cfg.load_from = self.get_model_ckpt(model_ckpt)
        if pretrained and isinstance(pretrained, str):
            logger.info(f"Overriding cfg.load_from -> {pretrained}")
            cfg.load_from = pretrained  # Overriding by stage input
        if cfg.get("resume", False):
            cfg.resume_from = cfg.load_from

    @staticmethod
    def configure_hook(cfg, **kwargs):
        """Update cfg.custom_hooks based on cfg.custom_hook_options"""

        def update_hook(opt, custom_hooks, idx, hook):
            """Delete of update a custom hook"""
            if isinstance(opt, dict):
                if opt.get("_delete_", False):
                    # if option include _delete_=True, remove this hook from custom_hooks
                    logger.info(f"configure_hook: {hook['type']} is deleted")
                    del custom_hooks[idx]
                else:
                    logger.info(f"configure_hook: {hook['type']} is updated with {opt}")
                    hook.update(**opt)

        custom_hook_options = cfg.pop("custom_hook_options", {})
        # logger.info(f"configure_hook() {cfg.get('custom_hooks', [])} <- {custom_hook_options}")
        custom_hooks = cfg.get("custom_hooks", [])
        for idx, hook in enumerate(custom_hooks):
            for opt_key, opt in custom_hook_options.items():
                if hook["type"] == opt_key:
                    update_hook(opt, custom_hooks, idx, hook)

    @staticmethod
    def configure_samples_per_gpu(
        cfg: Config,
        subset: str,
        distributed: bool = False,
    ):

        dataloader_cfg = cfg.data.get(f"{subset}_dataloader", ConfigDict())
        samples_per_gpu = dataloader_cfg.get("samples_per_gpu", cfg.data.get("samples_per_gpu", 1))

        data_cfg = get_data_cfg(cfg, subset)
        dataset_len = len(data_cfg.otx_dataset)

        if distributed:
            dataset_len = dataset_len // dist.get_world_size()

        # set batch size as a total dataset
        # if it is smaller than total dataset
        if dataset_len < samples_per_gpu:
            dataloader_cfg.samples_per_gpu = dataset_len

        # drop the last batch if the last batch size is 1
        # batch size of 1 is a runtime error for training batch normalization layer
        if subset in ("train", "unlabeled") and dataset_len % samples_per_gpu == 1:
            dataloader_cfg.drop_last = True

        cfg.data[f"{subset}_dataloader"] = dataloader_cfg

    @staticmethod
    def configure_compat_cfg(
        cfg: Config,
    ):
        """Modify config to keep the compatibility."""

        def _configure_dataloader(cfg):
            """Consume all the global dataloader config and convert them
            to specific dataloader config as it would be deprecated in the future.
            """
            global_dataloader_cfg = {}
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
                dataloader_cfg = {**global_dataloader_cfg, **dataloader_cfg}
                cfg.data[f"{subset}_dataloader"] = dataloader_cfg

        _configure_dataloader(cfg)

    @staticmethod
    def configure_fp16_optimizer(cfg: Config, distributed: bool = False):
        """Configure Fp16OptimizerHook and Fp16SAMOptimizerHook."""

        fp16_config = cfg.pop("fp16", None)
        if fp16_config is not None:
            optim_type = cfg.optimizer_config.get("type", "OptimizerHook")
            opts: Dict[str, Any] = dict(
                distributed=distributed,
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

    @staticmethod
    def configure_unlabeled_dataloader(cfg: Config, distributed: bool = False):
        if "unlabeled" in cfg.data:
            task_lib_module = importlib.import_module(f"{MODEL_TASK[cfg.model_task]}.datasets")
            dataset_builder = getattr(task_lib_module, "build_dataset")
            dataloader_builder = getattr(task_lib_module, "build_dataloader")

            dataset = build_dataset(cfg, "unlabeled", dataset_builder, consume=True)
            unlabeled_dataloader = build_dataloader(
                dataset,
                cfg,
                "unlabeled",
                dataloader_builder,
                distributed=distributed,
                consume=True,
            )

            custom_hooks = cfg.get("custom_hooks", [])
            updated = False
            for custom_hook in custom_hooks:
                if custom_hook["type"] == "ComposedDataLoadersHook":
                    custom_hook["data_loaders"] = [*custom_hook["data_loaders"], unlabeled_dataloader]
                    updated = True
            if not updated:
                custom_hooks.append(
                    ConfigDict(
                        type="ComposedDataLoadersHook",
                        data_loaders=unlabeled_dataloader,
                    )
                )
            cfg.custom_hooks = custom_hooks

    @staticmethod
    def get_model_meta(cfg):
        ckpt_path = cfg.get("load_from", None)
        meta = {}
        if ckpt_path:
            ckpt = CheckpointLoader.load_checkpoint(ckpt_path, map_location="cpu")
            meta = ckpt.get("meta", {})
        return meta

    @staticmethod
    def get_data_cfg(cfg, subset):
        assert subset in ["train", "val", "test"], f"Unknown subset:{subset}"
        if "dataset" in cfg.data[subset]:  # Concat|RepeatDataset
            dataset = cfg.data[subset].dataset
            while hasattr(dataset, "dataset"):
                dataset = dataset.dataset
            return dataset
        else:
            return cfg.data[subset]

    @staticmethod
    def get_data_classes(cfg):
        data_classes = []
        train_cfg = Stage.get_data_cfg(cfg, "train")
        if "data_classes" in train_cfg:
            data_classes = list(train_cfg.pop("data_classes", []))
        elif "classes" in train_cfg:
            data_classes = list(train_cfg.classes)
        return data_classes

    @staticmethod
    def get_model_classes(cfg):
        """Extract trained classes info from checkpoint file.
        MMCV-based models would save class info in ckpt['meta']['CLASSES']
        For other cases, try to get the info from cfg.model.classes (with pop())
        - Which means that model classes should be specified in model-cfg for
          non-MMCV models (e.g. OMZ models)
        """
        classes = []
        meta = Stage.get_model_meta(cfg)
        # for MPA classification legacy compatibility
        classes = meta.get("CLASSES", [])
        classes = meta.get("classes", classes)
        if classes is None:
            classes = []

        if len(classes) == 0:
            ckpt_path = cfg.get("load_from", None)
            if ckpt_path:
                classes = Stage.read_label_schema(ckpt_path)
        if len(classes) == 0:
            classes = cfg.model.pop("classes", cfg.pop("model_classes", []))
        return classes

    @staticmethod
    def get_model_ckpt(ckpt_path, new_path=None):
        ckpt = CheckpointLoader.load_checkpoint(ckpt_path, map_location="cpu")
        if "model" in ckpt:
            ckpt = ckpt["model"]
            if not new_path:
                new_path = ckpt_path[:-3] + "converted.pth"
            torch.save(ckpt, new_path)
            return new_path
        else:
            return ckpt_path

    @staticmethod
    def read_label_schema(ckpt_path, name_only=True, file_name="label_schema.json"):
        serialized_label_schema = []
        if any(ckpt_path.endswith(extension) for extension in (".xml", ".bin", ".pth")):
            label_schema_path = osp.join(osp.dirname(ckpt_path), file_name)
            if osp.exists(label_schema_path):
                with open(label_schema_path, encoding="UTF-8") as read_file:
                    serialized_label_schema = json.load(read_file)
        if serialized_label_schema:
            if name_only:
                all_classes = [labels["name"] for labels in serialized_label_schema["all_labels"].values()]
            else:
                all_classes = serialized_label_schema
        else:
            all_classes = []
        return all_classes

    @staticmethod
    def set_inference_progress_callback(model, cfg):
        # InferenceProgressCallback (Time Monitor enable into Infer task)
        time_monitor = None
        if cfg.get("custom_hooks", None):
            time_monitor = [hook.time_monitor for hook in cfg.custom_hooks if hook.type == "OTXProgressHook"]
            time_monitor = time_monitor[0] if time_monitor else None
        if time_monitor is not None:

            def pre_hook(module, input):
                time_monitor.on_test_batch_begin(None, None)

            def hook(module, input, output):
                time_monitor.on_test_batch_end(None, None)

            model.register_forward_pre_hook(pre_hook)
            model.register_forward_hook(hook)

    @classmethod
    def build_model(
        cls,
        cfg: Config,
        model_builder: Optional[Callable] = None,
        *,
        fp16: bool = False,
        **kwargs,
    ) -> torch.nn.Module:
        if model_builder is None:
            model_builder = cls.MODEL_BUILDER
        assert model_builder is not None
        model = model_builder(cfg, **kwargs)
        if bool(fp16):
            wrap_fp16_model(model)
        return model

    def _get_feature_module(self, model):
        return model
