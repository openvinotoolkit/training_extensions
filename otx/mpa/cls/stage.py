# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import copy

import numpy as np
import torch
from mmcv import ConfigDict, build_from_cfg

from otx.algorithms.classification.adapters.mmcls.utils.builder import build_classifier
from otx.mpa.stage import Stage
from otx.mpa.utils.config_utils import recursively_update_cfg, update_or_add_custom_hook
from otx.mpa.utils.logger import get_logger

logger = get_logger()


class ClsStage(Stage):
    MODEL_BUILDER = build_classifier

    def configure(self, model_cfg, model_ckpt, data_cfg, training=True, **kwargs):  # noqa: C901
        """Create MMCV-consumable config from given inputs"""
        logger.info(f"configure: training={training}")

        # Recipe + model
        cfg = self.cfg
        self.configure_model(cfg, model_cfg, training, **kwargs)
        self.configure_ckpt(cfg, model_ckpt, kwargs.get("pretrained", None))
        self.configure_data(cfg, data_cfg, training, **kwargs)
        self.configure_task(cfg, training, **kwargs)
        return cfg

    def configure_model(self, cfg, model_cfg, training, **kwargs):
        if model_cfg:
            if hasattr(cfg, "model"):
                cfg.merge_from_dict(model_cfg._cfg_dict)
            else:
                cfg.model = copy.deepcopy(model_cfg.model)

        cfg.model_task = cfg.model.pop("task", "classification")
        if cfg.model_task != "classification":
            raise ValueError(f"Given model_cfg ({model_cfg.filename}) is not supported by classification recipe")

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
        self.configure_in_channel(cfg)
        self.configure_topk(cfg)

    @staticmethod
    def configure_in_channel(cfg):
        configure_required = False
        if cfg.model.get("neck") is not None:
            if cfg.model.neck.get("in_channels") is not None and cfg.model.neck.in_channels <= 0:
                configure_required = True
        if not configure_required and cfg.model.get("head") is not None:
            if cfg.model.head.get("in_channels") is not None and cfg.model.head.in_channels <= 0:
                configure_required = True
        if not configure_required:
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

    @staticmethod
    def configure_topk(cfg):
        if cfg.model.head.get("topk", False) and isinstance(cfg.model.head.topk, tuple):
            cfg.model.head.topk = (1,) if cfg.model.head.num_classes < 5 else (1, 5)
            if cfg.model.get("multilabel", False) or cfg.model.get("hierarchical", False):
                cfg.model.head.pop("topk", None)

    def configure_ckpt(self, cfg, model_ckpt, pretrained):
        # Checkpoint
        if model_ckpt:
            cfg.load_from = self.get_model_ckpt(model_ckpt)
        if pretrained and isinstance(pretrained, str):
            logger.info(f"Overriding cfg.load_from -> {pretrained}")
            cfg.load_from = pretrained  # Overriding by stage input
        if cfg.get("resume", False):
            cfg.resume_from = cfg.load_from
        if cfg.get("load_from", None) and cfg.model.backbone.get("pretrained", None):
            cfg.model.backbone.pretrained = None

    def configure_data(self, cfg, data_cfg, training, **kwargs):
        # Data
        if data_cfg:
            cfg.merge_from_dict(data_cfg)
        Stage.configure_data(cfg, training, **kwargs)

    def configure_task(self, cfg, training, **kwargs):
        self.configure_classes(cfg)

    def configure_classes(self, cfg):
        model_classes, data_classes = [], []
        self.model_meta = self.get_model_meta(cfg)
        train_data_cfg = Stage.get_data_cfg(cfg, "train")
        if isinstance(train_data_cfg, list):
            train_data_cfg = train_data_cfg[0]

        model_classes = Stage.get_model_classes(cfg)
        data_classes = Stage.get_data_classes(cfg)

        if cfg.get("model_classes", []):
            cfg.model.head.num_classes = len(cfg.model_classes)
        elif model_classes:
            cfg.model.head.num_classes = len(model_classes)
        elif data_classes:
            cfg.model.head.num_classes = len(data_classes)
        self.model_meta["CLASSES"] = model_classes

        if not train_data_cfg.get("new_classes", False):  # when train_data_cfg doesn't have 'new_classes' key
            new_classes = np.setdiff1d(data_classes, model_classes).tolist()
            train_data_cfg["new_classes"] = new_classes
        self.model_classes = model_classes
        self.data_classes = data_classes
