# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.algorithms.detection.adapters.mmdet.utils.builder import build_detector
from otx.mpa.stage import Stage
from otx.mpa.utils.config_utils import recursively_update_cfg
from otx.mpa.utils.logger import get_logger

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
        self.configure_data(cfg, data_cfg, training, **kwargs)
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

        # OMZ-plugin
        if cfg.model.backbone.type == "OmzBackboneDet":
            ir_path = kwargs.get("ir_path")
            if not ir_path:
                raise RuntimeError("OMZ model needs OpenVINO bin/xml files.")
            cfg.model.backbone.model_path = ir_path
            if cfg.model.type == "SingleStageDetector":
                cfg.model.bbox_head.model_path = ir_path
            elif cfg.model.type == "FasterRCNN":
                cfg.model.rpn_head.model_path = ir_path
            else:
                raise NotImplementedError(f"Unknown model type - {cfg.model.type}")

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

    def configure_ckpt(self, cfg, model_ckpt, pretrained):
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

    def configure_data(self, cfg, data_cfg, training, **kwargs):  # noqa: C901
        """Patch cfg.data.
        Merge cfg and data_cfg
        Match cfg.data.train.type to super_type
        Patch for unlabeled data path ==> This may be moved to SemiDetectionStage
        """
        if data_cfg:
            cfg.merge_from_dict(data_cfg)

        Stage.configure_data(cfg, training, **kwargs)
        super_type = cfg.data.train.pop("super_type", None)
        if super_type:
            cfg.data.train.org_type = cfg.data.train.type
            cfg.data.train.type = super_type
        if training:
            if "dataset" in cfg.data.train:
                train_cfg = self.get_data_cfg(cfg, "train")
                if cfg.data.train.get("otx_dataset", None) is not None:
                    train_cfg.otx_dataset = cfg.data.train.pop("otx_dataset")
                if cfg.data.train.get("labels", None) is not None:
                    train_cfg.labels = cfg.data.train.get("labels")
                if cfg.data.train.get("data_classes", None) is not None:
                    train_cfg.data_classes = cfg.data.train.pop("data_classes")
                if cfg.data.train.get("new_classes", None) is not None:
                    train_cfg.new_classes = cfg.data.train.pop("new_classes")

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
        """Patch config to support training algorithm.

        This should be implemented each algorithm
        """
        pass
