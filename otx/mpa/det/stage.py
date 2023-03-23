# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.utils import ConfigDict

from otx.algorithms.common.adapters.mmcv.utils.config_utils import (
    recursively_update_cfg,
    update_or_add_custom_hook,
)
from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.detection.adapters.mmdet.utils.builder import build_detector
from otx.mpa.stage import Stage

logger = get_logger()


class DetectionStage(Stage):
    """Patch config to support otx train."""

    MODEL_BUILDER = build_detector

    def configure(self, model_cfg, model_ckpt, data_cfg, training=True, **kwargs):
        """Create MMCV-consumable config from given inputs"""
        logger.info(f"configure!: training={training}")

        cfg = self.cfg
        self.configure_model(cfg, model_cfg, training, **kwargs)
        self.configure_ckpt(cfg, model_ckpt, kwargs.get("pretrained", None))
        self.configure_data(cfg, training, data_cfg, **kwargs)
        self.configure_regularization(cfg, training)
        self.configure_hyperparams(cfg, training, **kwargs)
        self.configure_task(cfg, training, **kwargs)
        self.configure_hook(cfg)
        return cfg

    def configure_model(self, cfg, model_cfg, training, **kwargs):  # noqa: C901
        """Patch config's model.
        Replace cfg.model to model_cfg
        Change model type to super type
        Patch for OMZ backbones
        """
        if model_cfg:
            if hasattr(model_cfg, "model"):
                cfg.merge_from_dict(model_cfg._cfg_dict)
            else:
                raise ValueError(
                    "Unexpected config was passed through 'model_cfg'. "
                    "it should have 'model' attribute in the config"
                )
            cfg.model_task = cfg.model.pop("task", "detection")
            if cfg.model_task != "detection":
                raise ValueError(f"Given model_cfg ({model_cfg.filename}) is not supported by detection recipe")

        super_type = cfg.model.pop("super_type", None)
        if super_type:
            cfg.model.arch_type = cfg.model.type
            cfg.model.type = super_type

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

    def configure_data(self, cfg, training, data_cfg, **kwargs):  # noqa: C901
        """Patch cfg.data.
        Merge cfg and data_cfg
        Match cfg.data.train.type to super_type
        Patch for unlabeled data path ==> This may be moved to SemiDetectionStage
        """
        if data_cfg:
            cfg.merge_from_dict(data_cfg)

        super().configure_data(cfg, training, **kwargs)
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

    def configure_hyperparams(self, cfg, training, **kwargs):
        """Patch optimization hyparms such as batch size, learning rate."""
        if "hyperparams" in cfg:
            hyperparams = kwargs.get("hyperparams", None)
            if hyperparams is not None:
                bs = hyperparams.get("bs", None)
                if bs is not None:
                    cfg.data.samples_per_gpu = bs

                lr = hyperparams.get("lr", None)
                if lr is not None:
                    cfg.optimizer.lr = lr

    def configure_task(self, cfg, training, **kwargs):
        """Patch config to support training algorithm."""
        if "task_adapt" in cfg:
            logger.info(f"task config!!!!: training={training}")
            self.task_adapt_type = cfg["task_adapt"].get("type", None)
            self.task_adapt_op = cfg["task_adapt"].get("op", "REPLACE")
            self.configure_classes(cfg)

            if self.data_classes != self.model_classes:
                self.configure_task_data_pipeline(cfg)
            # TODO[JAEGUK]: configure_anchor is not working
            if cfg["task_adapt"].get("use_mpa_anchor", False):
                self.configure_anchor(cfg)
            if self.task_adapt_type == "mpa":
                self.configure_bbox_head(cfg)
                self.configure_ema(cfg)
            else:
                src_data_cfg = self.get_data_cfg(cfg, "train")
                src_data_cfg.pop("old_new_indices", None)

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
        # Trying to alter class indices of training data according to model class order
        tr_data_cfg = self.get_data_cfg(cfg, "train")
        class_adapt_cfg = dict(type="AdaptClassLabels", src_classes=self.data_classes, dst_classes=self.model_classes)
        pipeline_cfg = tr_data_cfg.pipeline
        for i, op in enumerate(pipeline_cfg):
            if op["type"] == "LoadAnnotations":  # insert just after this op
                op_next_ann = pipeline_cfg[i + 1] if i + 1 < len(pipeline_cfg) else {}
                if op_next_ann.get("type", "") == class_adapt_cfg["type"]:
                    op_next_ann.update(class_adapt_cfg)
                else:
                    pipeline_cfg.insert(i + 1, class_adapt_cfg)
                break

    def configure_anchor(self, cfg, proposal_ratio=None):
        if cfg.model.type in ["SingleStageDetector", "CustomSingleStageDetector"]:
            anchor_cfg = cfg.model.bbox_head.anchor_generator
            if anchor_cfg.type == "SSDAnchorGeneratorClustered":
                cfg.model.bbox_head.anchor_generator.pop("input_size", None)

    def configure_bbox_head(self, cfg):
        """Patch bbox head in detector for class incremental learning.
        Most of patching are related with hyper-params in focal loss
        """
        if cfg.get("task", "detection") == "detection":
            bbox_head = cfg.model.bbox_head
        else:
            bbox_head = cfg.model.roi_head.bbox_head

        alpha, gamma = 0.25, 2.0
        if bbox_head.type in ["SSDHead", "CustomSSDHead"]:
            gamma = 1 if cfg["task_adapt"].get("efficient_mode", False) else 2
            bbox_head.type = "CustomSSDHead"
            bbox_head.loss_cls = ConfigDict(
                type="FocalLoss",
                loss_weight=1.0,
                gamma=gamma,
                reduction="none",
            )
        elif bbox_head.type in ["ATSSHead"]:
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
