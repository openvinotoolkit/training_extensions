"""Base configurer for mmdet config."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import importlib
import json
import os
from typing import Any, Dict

import numpy as np
import torch
from mmcv.runner import CheckpointLoader
from mmcv.utils import Config, ConfigDict
from torch import distributed as dist

from otx.algorithms.common.adapters.mmcv.utils import (
    align_data_config_with_recipe,
    build_dataloader,
    build_dataset,
    patch_adaptive_interval_training,
    patch_default_config,
    patch_early_stopping,
    patch_fp16,
    patch_persistent_workers,
    patch_runner,
)
from otx.algorithms.common.adapters.mmcv.utils.config_utils import (
    recursively_update_cfg,
    update_or_add_custom_hook,
)
from otx.algorithms.common.utils import append_dist_rank_suffix
from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.detection.adapters.mmdet.utils import (
    cluster_anchors,
    patch_datasets,
    patch_evaluation,
    should_cluster_anchors,
)

logger = get_logger()


# pylint: disable=too-many-public-methods
class DetectionConfigurer:
    """Patch config to support otx train."""

    def __init__(self):
        self.task_adapt_type = None
        self.task_adapt_op = "REPLACE"
        self.org_model_classes = []
        self.model_classes = []
        self.data_classes = []

    # pylint: disable=too-many-arguments
    def configure(
        self,
        cfg,
        train_dataset,
        model_ckpt,
        data_cfg,
        training=True,
        subset="train",
        ir_options=None,
        data_classes=None,
        model_classes=None,
    ):
        """Create MMCV-consumable config from given inputs."""
        logger.info(f"configure!: training={training}")

        self.configure_base(cfg, data_cfg, data_classes, model_classes)
        self.configure_device(cfg, training)
        self.configure_model(cfg, ir_options)
        self.configure_ckpt(cfg, model_ckpt)
        self.configure_data(cfg, training, data_cfg)
        self.configure_regularization(cfg, training)
        self.configure_task(cfg, train_dataset, training)
        self.configure_hook(cfg)
        self.configure_samples_per_gpu(cfg, subset)
        self.configure_fp16_optimizer(cfg)
        self.configure_compat_cfg(cfg)
        return cfg

    def configure_base(self, cfg, data_cfg, data_classes, model_classes):
        """Basic configuration work for recipe.

        Patchings in this function are handled task level previously
        This function might need to be re-orgianized
        """

        options_for_patch_datasets = {"type": "OTXDetDataset"}

        patch_default_config(cfg)
        patch_runner(cfg)
        patch_datasets(
            cfg,
            **options_for_patch_datasets,
        )  # for OTX compatibility
        patch_evaluation(cfg)  # for OTX compatibility
        patch_fp16(cfg)
        patch_adaptive_interval_training(cfg)
        patch_early_stopping(cfg)
        patch_persistent_workers(cfg)

        if data_cfg is not None:
            align_data_config_with_recipe(data_cfg, cfg)

        # update model config -> model label schema
        cfg["model_classes"] = model_classes
        if data_classes is not None:
            train_data_cfg = self.get_data_cfg(data_cfg, "train")
            train_data_cfg["data_classes"] = data_classes
            new_classes = np.setdiff1d(data_classes, model_classes).tolist()
            train_data_cfg["new_classes"] = new_classes

    def configure_model(self, cfg, ir_options):  # noqa: C901
        """Patch config's model.

        Change model type to super type
        Patch for OMZ backbones
        """

        if ir_options is None:
            ir_options = {"ir_model_path": None, "ir_weight_path": None, "ir_weight_init": False}

        cfg.model_task = cfg.model.pop("task", "detection")
        if cfg.model_task != "detection":
            raise ValueError(f"Given cfg ({cfg.filename}) is not supported by detection recipe")

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

    def configure_data(self, cfg, training, data_cfg):  # noqa: C901
        """Patch cfg.data.

        Merge cfg and data_cfg
        Match cfg.data.train.type to super_type
        Patch for unlabeled data path ==> This may be moved to SemiDetectionConfigurer
        """
        if data_cfg:
            cfg.merge_from_dict(data_cfg)

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

            # pylint: disable=too-many-branches, too-many-nested-blocks
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
        super_type = cfg.data.train.pop("super_type", None)
        if super_type:
            cfg.data.train.org_type = cfg.data.train.type
            cfg.data.train.type = super_type

    def configure_regularization(self, cfg, training):  # noqa: C901
        """Patch regularization parameters."""
        if training:
            if cfg.model.get("l2sp_weight", 0.0) > 0.0:
                logger.info("regularization config!!!!")

                # Checkpoint
                l2sp_ckpt = cfg.model.get("l2sp_ckpt", None)
                if l2sp_ckpt is None:
                    if "pretrained" in cfg.model:
                        l2sp_ckpt = cfg.model.pretrained
                    if cfg.load_from:
                        l2sp_ckpt = cfg.load_from
                cfg.model.l2sp_ckpt = l2sp_ckpt

                # Disable weight decay
                if "weight_decay" in cfg.optimizer:
                    cfg.optimizer.weight_decay = 0.0

    def configure_task(self, cfg, train_dataset, training):
        """Patch config to support training algorithm."""
        if "task_adapt" in cfg:
            logger.info(f"task config!!!!: training={training}")
            self.task_adapt_type = cfg["task_adapt"].get("type", None)
            self.task_adapt_op = cfg["task_adapt"].get("op", "REPLACE")
            self.configure_classes(cfg)

            if self.data_classes != self.model_classes:
                self.configure_task_data_pipeline(cfg)
            if cfg["task_adapt"].get("use_mpa_anchor", False):
                self.configure_anchor(cfg, train_dataset)
            if self.task_adapt_type == "mpa":
                self.configure_bbox_head(cfg)
                self.configure_ema(cfg)
            else:
                src_data_cfg = self.get_data_cfg(cfg, "train")
                src_data_cfg.pop("old_new_indices", None)

    # pylint: disable=too-many-branches
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

        if self.task_adapt_type == "mpa":
            data_classes = model_classes
        cfg.task_adapt.final = model_classes
        cfg.model.task_adapt = ConfigDict(
            src_classes=org_model_classes,
            dst_classes=model_classes,
        )

        # Model architecture
        head_names = ("mask_head", "bbox_head", "segm_head")
        num_classes = len(model_classes)
        if "roi_head" in cfg.model:
            # For Faster-RCNNs
            for head_name in head_names:
                if head_name in cfg.model.roi_head:
                    if isinstance(cfg.model.roi_head[head_name], list):
                        for head in cfg.model.roi_head[head_name]:
                            head.num_classes = num_classes
                    else:
                        cfg.model.roi_head[head_name].num_classes = num_classes
        else:
            # For other architectures (including SSD)
            for head_name in head_names:
                if head_name in cfg.model:
                    cfg.model[head_name].num_classes = num_classes

        # Eval datasets
        if cfg.get("task", "detection") == "detection":
            eval_types = ["val", "test"]
            for eval_type in eval_types:
                if cfg.data[eval_type]["type"] == "TaskAdaptEvalDataset":
                    cfg.data[eval_type]["model_classes"] = model_classes
                else:
                    # Wrap original dataset config
                    org_type = cfg.data[eval_type]["type"]
                    cfg.data[eval_type]["type"] = "TaskAdaptEvalDataset"
                    cfg.data[eval_type]["org_type"] = org_type
                    cfg.data[eval_type]["model_classes"] = model_classes

        self.org_model_classes = org_model_classes
        self.model_classes = model_classes
        self.data_classes = data_classes

    def configure_task_data_pipeline(self, cfg):
        """Trying to alter class indices of training data according to model class order."""
        tr_data_cfg = self.get_data_cfg(cfg, "train")
        class_adapt_cfg = dict(type="AdaptClassLabels", src_classes=self.data_classes, dst_classes=self.model_classes)
        pipeline_cfg = tr_data_cfg.pipeline
        for i, operation in enumerate(pipeline_cfg):
            if operation["type"] == "LoadAnnotations":  # insert just after this operation
                op_next_ann = pipeline_cfg[i + 1] if i + 1 < len(pipeline_cfg) else {}
                if op_next_ann.get("type", "") == class_adapt_cfg["type"]:
                    op_next_ann.update(class_adapt_cfg)
                else:
                    pipeline_cfg.insert(i + 1, class_adapt_cfg)
                break

    def configure_anchor(self, cfg, train_dataset):
        """Patch anchor settings for single stage detector."""
        if cfg.model.type in ["SingleStageDetector", "CustomSingleStageDetector"]:
            anchor_cfg = cfg.model.bbox_head.anchor_generator
            if anchor_cfg.type == "SSDAnchorGeneratorClustered":
                cfg.model.bbox_head.anchor_generator.pop("input_size", None)
        if should_cluster_anchors(cfg) and train_dataset is not None:
            cluster_anchors(cfg, train_dataset)

    def configure_bbox_head(self, cfg):
        """Patch bbox head in detector for class incremental learning.

        Most of patching are related with hyper-params in focal loss
        """
        if cfg.get("task", "detection") == "detection":
            bbox_head = cfg.model.bbox_head
        else:
            bbox_head = cfg.model.roi_head.bbox_head

        alpha, gamma = 0.25, 2.0
        if bbox_head.type in ["ATSSHead"]:
            gamma = 3 if cfg["task_adapt"].get("efficient_mode", False) else 4.5
            bbox_head.loss_cls.gamma = gamma
        elif bbox_head.type in ["VFNetHead", "CustomVFNetHead"]:
            alpha = 0.75
            gamma = 1 if cfg["task_adapt"].get("efficient_mode", False) else 2
        # TODO Move this part
        # This is not related with patching bbox head
        elif bbox_head.type in ["YOLOXHead", "CustomYOLOXHead"]:
            if cfg.data.train.type == "MultiImageMixDataset":
                self.add_yolox_hooks(cfg)

        if cfg.get("ignore", False):
            bbox_head.loss_cls = ConfigDict(
                type="CrossSigmoidFocalLoss",
                use_sigmoid=True,
                num_classes=len(self.model_classes),
                alpha=alpha,
                gamma=gamma,
            )

    @staticmethod
    def configure_ema(cfg):
        """Patch ema settings."""
        adaptive_ema = cfg.get("adaptive_ema", {})
        if adaptive_ema:
            update_or_add_custom_hook(
                cfg,
                ConfigDict(
                    type="CustomModelEMAHook",
                    priority="ABOVE_NORMAL",
                    resume_from=cfg.get("resume_from"),
                    **adaptive_ema,
                ),
            )
        else:
            update_or_add_custom_hook(
                cfg,
                ConfigDict(type="EMAHook", priority="ABOVE_NORMAL", resume_from=cfg.get("resume_from"), momentum=0.1),
            )

    @staticmethod
    def add_yolox_hooks(cfg):
        """Update yolox hooks."""
        update_or_add_custom_hook(
            cfg,
            ConfigDict(
                type="YOLOXModeSwitchHook",
                num_last_epochs=15,
                priority=48,
            ),
        )
        update_or_add_custom_hook(
            cfg,
            ConfigDict(
                type="SyncNormHook",
                num_last_epochs=15,
                interval=1,
                priority=48,
            ),
        )

    # Functions below are come from base stage
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

    @staticmethod
    def get_model_classes(cfg):
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

        def read_label_schema(ckpt_path, name_only=True, file_name="label_schema.json"):
            serialized_label_schema = []
            if any(ckpt_path.endswith(extension) for extension in (".xml", ".bin", ".pth")):
                label_schema_path = os.path.join(os.path.dirname(ckpt_path), file_name)
                if os.path.exists(label_schema_path):
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
                classes = read_label_schema(ckpt_path)
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
        assert subset in ["train", "val", "test"], f"Unknown subset:{subset}"
        if "dataset" in cfg.data[subset]:  # Concat|RepeatDataset
            dataset = cfg.data[subset].dataset
            while hasattr(dataset, "dataset"):
                dataset = dataset.dataset
            return dataset
        return cfg.data[subset]

    @staticmethod
    def configure_hook(cfg):
        """Update cfg.custom_hooks based on cfg.custom_hook_options."""

        def update_hook(opt, custom_hooks, idx, hook):
            """Delete of update a custom hook."""
            if isinstance(opt, dict):
                if opt.get("_delete_", False):
                    # if option include _delete_=True, remove this hook from custom_hooks
                    logger.info(f"configure_hook: {hook['type']} is deleted")
                    del custom_hooks[idx]
                else:
                    logger.info(f"configure_hook: {hook['type']} is updated with {opt}")
                    hook.update(**opt)

        hook_cfg = ConfigDict(type="LoggerReplaceHook")
        update_or_add_custom_hook(cfg, hook_cfg)

        custom_hook_options = cfg.pop("custom_hook_options", {})
        # logger.info(f"configure_hook() {cfg.get('custom_hooks', [])} <- {custom_hook_options}")
        custom_hooks = cfg.get("custom_hooks", [])
        for idx, hook in enumerate(custom_hooks):
            for opt_key, opt in custom_hook_options.items():
                if hook["type"] == opt_key:
                    update_hook(opt, custom_hooks, idx, hook)

    def configure_device(self, cfg, training):
        """Setting device for training and inference."""
        cfg.distributed = False
        if torch.distributed.is_initialized():
            cfg.gpu_ids = [int(os.environ["LOCAL_RANK"])]
            if training:  # TODO multi GPU is available only in training. Evaluation needs to be supported later.
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

    def configure_samples_per_gpu(
        self,
        cfg: Config,
        subset: str,
    ):
        """Settings samples_per_gpu for training and inference."""

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

            # drop the last batch if the last batch size is 1
            # batch size of 1 is a runtime error for training batch normalization layer
            if subset in ("train", "unlabeled") and dataset_len % samples_per_gpu == 1:
                dataloader_cfg["drop_last"] = True

            cfg.data[f"{subset}_dataloader"] = dataloader_cfg

    @staticmethod
    def configure_fp16_optimizer(cfg: Config):
        """Configure Fp16OptimizerHook and Fp16SAMOptimizerHook."""

        fp16_config = cfg.pop("fp16", None)
        if fp16_config is not None:
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

    @staticmethod
    def configure_distributed(cfg):
        """Patching for distributed training."""
        if hasattr(cfg, "dist_params") and cfg.dist_params.get("linear_scale_lr", False):
            new_lr = dist.get_world_size() * cfg.optimizer.lr
            logger.info(
                f"enabled linear scaling rule to the learning rate. \
                changed LR from {cfg.optimizer.lr} to {new_lr}"
            )
            cfg.optimizer.lr = new_lr

    @staticmethod
    def configure_compat_cfg(
        cfg: Config,
    ):
        """Modify config to keep the compatibility."""

        def _configure_dataloader(cfg):
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
                dataloader_cfg = Config(cfg_dict={**global_dataloader_cfg, **dataloader_cfg})
                cfg.data[f"{subset}_dataloader"] = dataloader_cfg

        _configure_dataloader(cfg)


class IncrDetectionConfigurer(DetectionConfigurer):
    """Patch config to support incremental learning for object detection."""

    def configure_task(self, cfg, train_dataset, training):
        """Patch config to support incremental learning."""
        super().configure_task(cfg, train_dataset, training)
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


class SemiSLDetectionConfigurer(DetectionConfigurer):
    """Patch config to support semi supervised learning for object detection."""

    def configure_data(self, cfg, training, data_cfg):
        """Patch cfg.data."""
        super().configure_data(cfg, training, data_cfg)
        # Set unlabeled data hook
        if training:
            if cfg.data.get("unlabeled", False) and cfg.data.unlabeled.get("otx_dataset", False):
                if len(cfg.data.unlabeled.get("pipeline", [])) == 0:
                    cfg.data.unlabeled.pipeline = cfg.data.train.pipeline.copy()
                self.configure_unlabeled_dataloader(cfg)

    def configure_task(self, cfg, train_dataset, training):
        """Patch config to support training algorithm."""
        logger.info(f"Semi-SL task config!!!!: training={training}")
        if "task_adapt" in cfg:
            self.task_adapt_type = cfg["task_adapt"].get("type", None)
            self.task_adapt_op = cfg["task_adapt"].get("op", "REPLACE")
            self.configure_classes(cfg)

            if self.data_classes != self.model_classes:
                self.configure_task_data_pipeline(cfg)
            if cfg["task_adapt"].get("use_mpa_anchor", False):
                self.configure_anchor(cfg, train_dataset)
            if self.task_adapt_type == "mpa":
                self.configure_bbox_head(cfg)
            else:
                src_data_cfg = self.get_data_cfg(cfg, "train")
                src_data_cfg.pop("old_new_indices", None)

    @staticmethod
    def configure_unlabeled_dataloader(cfg: Config):
        """Patch for unlabled dataloader."""

        model_task = {"classification": "mmcls", "detection": "mmdet", "segmentation": "mmseg"}
        if "unlabeled" in cfg.data:
            task_lib_module = importlib.import_module(f"{model_task[cfg.model_task]}.datasets")
            dataset_builder = getattr(task_lib_module, "build_dataset")
            dataloader_builder = getattr(task_lib_module, "build_dataloader")

            dataset = build_dataset(cfg, "unlabeled", dataset_builder, consume=True)
            unlabeled_dataloader = build_dataloader(
                dataset,
                cfg,
                "unlabeled",
                dataloader_builder,
                distributed=cfg.distributed,
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
