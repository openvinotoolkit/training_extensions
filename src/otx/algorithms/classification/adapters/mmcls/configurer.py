"""Base configurer for mmdet config."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import importlib
from typing import Optional

import torch
from mmcv import build_from_cfg
from mmcv.utils import Config, ConfigDict

from otx.algorithms import TRANSFORMER_BACKBONES
from otx.algorithms.classification.adapters.mmcls.utils import (
    patch_datasets,
    patch_evaluation,
)
from otx.algorithms.common.adapters.mmcv.configurer import BaseConfigurer
from otx.algorithms.common.adapters.mmcv.utils import (
    build_dataloader,
    build_dataset,
)
from otx.algorithms.common.adapters.mmcv.utils.config_utils import (
    InputSizeManager,
    get_configured_input_size,
    recursively_update_cfg,
    update_or_add_custom_hook,
)
from otx.algorithms.common.configs.configuration_enums import InputSizePreset
from otx.algorithms.common.utils.logger import get_logger

logger = get_logger()


# pylint: disable=too-many-public-methods
class ClassificationConfigurer(BaseConfigurer):
    """Patch config to support otx train."""

    # pylint: disable=too-many-arguments
    def configure(
        self,
        cfg,
        model_ckpt,
        data_cfg,
        ir_options=None,
        data_classes=None,
        model_classes=None,
        input_size: InputSizePreset = InputSizePreset.DEFAULT,
        **kwargs,
    ):
        """Create MMCV-consumable config from given inputs."""
        logger.info(f"configure!: training={self.training}")

        self.configure_base(cfg, data_cfg, data_classes, model_classes, **kwargs)
        self.configure_device(cfg)
        self.configure_ckpt(cfg, model_ckpt)
        self.configure_model(cfg, ir_options)
        self.configure_data(cfg, data_cfg)
        self.configure_task(cfg)
        self.configure_hook(cfg)
        self.configure_samples_per_gpu(cfg)
        self.configure_fp16(cfg)
        self.configure_compat_cfg(cfg)
        self.configure_input_size(cfg, input_size, model_ckpt)
        return cfg

    def configure_compatibility(self, cfg, **kwargs):
        """Configure for OTX compatibility with mmcls."""
        options_for_patch_datasets = kwargs.get("options_for_patch_datasets", {"type": "OTXClsDataset"})
        options_for_patch_evaluation = kwargs.get("options_for_patch_evaluation", {"task": "normal"})
        patch_datasets(cfg, **options_for_patch_datasets)
        patch_evaluation(cfg, **options_for_patch_evaluation)

    def configure_model(self, cfg, ir_options):  # noqa: C901
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
        cfg, input_size_config: InputSizePreset = InputSizePreset.DEFAULT, model_ckpt: Optional[str] = None
    ):
        """Change input size if necessary."""
        input_size = get_configured_input_size(input_size_config, model_ckpt)
        if input_size is None:
            return

        InputSizeManager(cfg.data).set_input_size(input_size)
        logger.info("Input size is changed to {}".format(input_size))


CLASS_INC_DATASET = [
    "OTXClsDataset",
    "OTXMultilabelClsDataset",
    "MPAHierarchicalClsDataset",
    "ClsTVDataset",
]
WEIGHT_MIX_CLASSIFIER = ["CustomImageClassifier"]


class IncrClassificationConfigurer(ClassificationConfigurer):
    """Patch config to support incremental learning for classification."""

    def configure_task(self, cfg):
        """Patch config to support incremental learning."""
        super().configure_task(cfg)
        if "task_adapt" in cfg and self.task_adapt_type == "mpa":
            self.configure_task_adapt_hook(cfg)

    def configure_task_adapt_hook(self, cfg):
        """Add TaskAdaptHook for sampler."""
        train_data_cfg = self.get_data_cfg(cfg, "train")
        if self.training:
            if train_data_cfg.type not in CLASS_INC_DATASET:
                logger.warning(f"Class Incremental Learning for {train_data_cfg.type} is not yet supported!")
            if "new_classes" not in train_data_cfg:
                logger.warning('"new_classes" should be defined for incremental learning w/ current model.')

            if cfg.model.type in WEIGHT_MIX_CLASSIFIER:
                cfg.model.task_adapt = ConfigDict(
                    src_classes=self.org_model_classes,
                    dst_classes=self.model_classes,
                )
            else:
                logger.warning(f"Weight mixing for {cfg.model.type} is not yet supported!")

            train_data_cfg.classes = self.model_classes

        if not cfg.model.get("multilabel", False) and not cfg.model.get("hierarchical", False):
            efficient_mode = cfg["task_adapt"].get("efficient_mode", True)
            sampler_type = "balanced"
            self.configure_loss(cfg)
        else:
            efficient_mode = cfg["task_adapt"].get("efficient_mode", False)
            sampler_type = "cls_incr"

        if len(set(self.org_model_classes) & set(self.model_classes)) == 0 or set(self.org_model_classes) == set(
            self.model_classes
        ):
            sampler_flag = False
        else:
            sampler_flag = True
        # Update Task Adapt Hook
        task_adapt_hook = ConfigDict(
            type="TaskAdaptHook",
            src_classes=self.org_model_classes,
            dst_classes=self.model_classes,
            model_type=cfg.model.type,
            sampler_flag=sampler_flag,
            sampler_type=sampler_type,
            efficient_mode=efficient_mode,
        )
        update_or_add_custom_hook(cfg, task_adapt_hook)

    def configure_loss(self, cfg):
        """Patch classification loss."""
        if len(set(self.org_model_classes) & set(self.model_classes)) == 0 or set(self.org_model_classes) == set(
            self.model_classes
        ):
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


class SemiSLClassificationConfigurer(ClassificationConfigurer):
    """Patch config to support semi supervised learning for classification."""

    def configure_data(self, cfg, data_cfg):
        """Patch cfg.data."""
        super().configure_data(cfg, data_cfg)
        # Set unlabeled data hook
        if self.training:
            if cfg.data.get("unlabeled", False) and cfg.data.unlabeled.get("otx_dataset", False):
                self.configure_unlabeled_dataloader(cfg)

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
