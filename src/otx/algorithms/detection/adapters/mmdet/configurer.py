"""Base configurer for mmdet config."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import importlib

from mmcv.utils import Config, ConfigDict

from otx.algorithms.common.adapters.mmcv.configurer import BaseConfigurer
from otx.algorithms.common.adapters.mmcv.utils import (
    build_dataloader,
    build_dataset,
)
from otx.algorithms.common.adapters.mmcv.utils.config_utils import (
    update_or_add_custom_hook,
)
from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.detection.adapters.mmdet.utils import (
    cluster_anchors,
    patch_datasets,
    patch_evaluation,
    should_cluster_anchors,
)

logger = get_logger()


# pylint: disable=too-many-public-methods
class DetectionConfigurer(BaseConfigurer):
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
        self.configure_model(cfg, ir_options, "detection")
        self.configure_ckpt(cfg, model_ckpt)
        self.configure_data(cfg, training, data_cfg)
        self.configure_regularization(cfg, training)
        self.configure_task(cfg, train_dataset, training)
        self.configure_hook(cfg)
        self.configure_samples_per_gpu(cfg, subset)
        self.configure_fp16_optimizer(cfg)
        self.configure_compat_cfg(cfg)
        return cfg

    def configure_compatibility(self, cfg, **kwargs):
        """Configure for OTX compatibility with mmdet."""
        options_for_patch_datasets = {"type": "OTXDetDataset"}
        patch_datasets(cfg, **options_for_patch_datasets)
        patch_evaluation(cfg)

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
        super().configure_task(cfg, training)
        if "task_adapt" in cfg:
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

    def configure_classes(self, cfg):
        """Patch classes for model and dataset."""
        super().configure_classes(cfg)
        self._configure_eval_dataset(cfg)

    def _configure_head(self, cfg):
        """Patch number of classes of head."""
        head_names = ("mask_head", "bbox_head", "segm_head")
        num_classes = len(self.model_classes)
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

    def _configure_eval_dataset(self, cfg):
        if cfg.get("task", "detection") == "detection":
            eval_types = ["val", "test"]
        for eval_type in eval_types:
            if cfg.data[eval_type]["type"] == "TaskAdaptEvalDataset":
                cfg.data[eval_type]["model_classes"] = self.model_classes
            else:
                # Wrap original dataset config
                org_type = cfg.data[eval_type]["type"]
                cfg.data[eval_type]["type"] = "TaskAdaptEvalDataset"
                cfg.data[eval_type]["org_type"] = org_type
                cfg.data[eval_type]["model_classes"] = self.model_classes

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
