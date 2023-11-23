"""Base configurer for mmdet config."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
from mmcv import build_from_cfg
from mmcv.utils import ConfigDict

from otx.algorithms import TRANSFORMER_BACKBONES
from otx.algorithms.classification.adapters.mmcls.utils import (
    patch_datasets,
    patch_evaluation,
)
from otx.algorithms.common.adapters.mmcv.clsincr_mixin import IncrConfigurerMixin
from otx.algorithms.common.adapters.mmcv.configurer import BaseConfigurer
from otx.algorithms.common.adapters.mmcv.semisl_mixin import SemiSLConfigurerMixin
from otx.algorithms.common.adapters.mmcv.utils.config_utils import (
    InputSizeManager,
    recursively_update_cfg,
    update_or_add_custom_hook,
)
from otx.utils.logger import get_logger

logger = get_logger()


# pylint: disable=too-many-public-methods
class ClassificationConfigurer(BaseConfigurer):
    """Patch config to support otx train."""

    def configure_data_pipeline(self, cfg, input_size, model_ckpt_path, **kwargs):
        """Configuration for data pipeline."""
        super().configure_data_pipeline(cfg, input_size, model_ckpt_path)
        options_for_patch_datasets = kwargs.get("options_for_patch_datasets", {"type": "OTXClsDataset"})
        patch_datasets(cfg, **options_for_patch_datasets)

    def configure_recipe(self, cfg, **kwargs):
        """Configuration for training recipe."""
        options_for_patch_evaluation = kwargs.get("options_for_patch_evaluation", {"task": "normal"})
        patch_evaluation(cfg, **options_for_patch_evaluation)
        super().configure_recipe(cfg)

    def configure_backbone(self, cfg, ir_options):  # noqa: C901
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

        # Hierarchical
        if cfg.model.get("hierarchical"):
            assert cfg.data.train.hierarchical_info == cfg.data.val.hierarchical_info == cfg.data.test.hierarchical_info
            cfg.model.head.hierarchical_info = cfg.data.train.hierarchical_info

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

        self.configure_in_channel(cfg)
        self.configure_topk(cfg)

    def _configure_head(self, cfg):
        """Patch nuber of classes of head.."""
        cfg.model.head.num_classes = len(self.model_classes)

    # pylint: disable=too-many-branches
    @staticmethod
    def configure_in_channel(cfg):
        """Return whether in_channels need patch."""
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

        if layer.__class__.__name__ in TRANSFORMER_BACKBONES and isinstance(output, (tuple, list)):
            # mmcls.VisionTransformer outputs Tuple[List[...]] and the last index of List is the final logit.
            _, output = output

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
        """Patch topk in case of num_classes is less than 5."""
        if cfg.model.head.get("topk", False) and isinstance(cfg.model.head.topk, tuple):
            cfg.model.head.topk = (1,) if cfg.model.head.num_classes < 5 else (1, 5)
            if cfg.model.get("multilabel", False) or cfg.model.get("hierarchical", False):
                cfg.model.head.pop("topk", None)

    @staticmethod
    def configure_input_size(
        cfg, input_size=Optional[Tuple[int, int]], model_ckpt_path: Optional[str] = None, training=True
    ):
        """Change input size if necessary."""
        if input_size is None:  # InputSizePreset.DEFAULT
            return

        manager = InputSizeManager(cfg)

        if input_size == (0, 0):  # InputSizePreset.AUTO
            if training:
                input_size = BaseConfigurer.adapt_input_size_to_dataset(cfg, manager, use_annotations=False)
            else:
                input_size = manager.get_trained_input_size(model_ckpt_path)
            if input_size is None:
                return

        manager.set_input_size(input_size)
        logger.info("Input size is changed to {}".format(input_size))


class IncrClassificationConfigurer(IncrConfigurerMixin, ClassificationConfigurer):
    """Patch config to support incremental learning for classification."""

    def configure_task(self, cfg, **kwargs):
        """Patch config to support incremental learning."""
        super().configure_task(cfg, **kwargs)
        if "task_adapt" in cfg and self.task_adapt_type == "default_task_adapt":
            self.configure_task_adapt_hook(cfg)
            if self._is_multiclass(cfg):
                self.configure_loss(cfg)

    def configure_loss(self, cfg):
        """Patch classification loss."""
        if not self.is_incremental():
            cfg.model.head.loss = dict(type="CrossEntropyLoss", loss_weight=1.0)
        else:
            cfg.model.head.loss = ConfigDict(
                type="IBLoss",
                num_classes=cfg.model.head.num_classes,
            )
            ib_loss_hook = ConfigDict(
                type="IBLossHook",
                dst_classes=self.model_classes,
            )
            update_or_add_custom_hook(cfg, ib_loss_hook)

    def get_sampler_type(self, cfg):
        """Return sampler type."""
        if self._is_multiclass(cfg):
            sampler_type = "balanced"
        else:
            sampler_type = "cls_incr"
        return sampler_type

    @staticmethod
    def _is_multiclass(cfg) -> bool:
        return not cfg.model.get("multilabel", False) and not cfg.model.get("hierarchical", False)


class SemiSLClassificationConfigurer(SemiSLConfigurerMixin, ClassificationConfigurer):
    """Patch config to support semi supervised learning for classification."""
