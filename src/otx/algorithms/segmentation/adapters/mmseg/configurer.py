"""Base configurer for mmseg config."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import importlib
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch
from mmcv.runner import CheckpointLoader
from mmcv.utils import Config, ConfigDict

from otx.algorithms.common.adapters.mmcv.configurer import BaseConfigurer
from otx.algorithms.common.adapters.mmcv.utils import (
    build_dataloader,
    build_dataset,
)
from otx.algorithms.common.adapters.mmcv.utils.config_utils import (
    InputSizeManager,
    get_configured_input_size,
    patch_color_conversion,
    remove_custom_hook,
    update_or_add_custom_hook,
)
from otx.algorithms.common.configs.configuration_enums import InputSizePreset
from otx.algorithms.common.utils import append_dist_rank_suffix
from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.segmentation.adapters.mmseg.models.heads import otx_head_factory

logger = get_logger()


# pylint: disable=too-many-public-methods
class SegmentationConfigurer(BaseConfigurer):
    """Patch config to support otx train."""

    # pylint: disable=too-many-arguments
    def configure(
        self,
        cfg: Config,
        model_ckpt: str,
        data_cfg: Config,
        ir_options: Optional[Config] = None,
        data_classes: Optional[List[str]] = None,
        model_classes: Optional[List[str]] = None,
        input_size: InputSizePreset = InputSizePreset.DEFAULT,
    ) -> Config:
        """Create MMCV-consumable config from given inputs."""
        logger.info(f"configure!: training={self.training}")

        self.configure_base(cfg, data_cfg, data_classes, model_classes)
        self.configure_device(cfg)
        self.configure_ckpt(cfg, model_ckpt)
        self.configure_model(cfg, ir_options)
        self.configure_data(cfg, data_cfg)
        self.configure_input_size(cfg, input_size, model_ckpt)
        self.configure_task(cfg)
        self.configure_hook(cfg)
        self.configure_samples_per_gpu(cfg)
        self.configure_fp16(cfg)
        self.configure_compat_cfg(cfg)
        return cfg

    def configure_compatibility(self, cfg, **kwargs):
        """Configure for OTX compatibility with mmseg."""
        patch_color_conversion(cfg)

    def configure_task(
        self,
        cfg: Config,
    ) -> None:
        """Patch config to support training algorithm."""
        super().configure_task(cfg)
        if "task_adapt" in cfg:
            self.configure_decode_head(cfg)

    def configure_classes(self, cfg):
        """Patch classes for model and dataset."""
        org_model_classes = self.get_model_classes(cfg)
        data_classes = self.get_data_classes(cfg)

        if "background" not in org_model_classes:
            org_model_classes = ["background"] + org_model_classes
        if "background" not in data_classes:
            data_classes = ["background"] + data_classes

        # Model classes
        if self.task_adapt_op == "REPLACE":
            if len(data_classes) == 1:  # background
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

    def configure_decode_head(self, cfg: Config) -> None:
        """Change to incremental loss (ignore mode) and substitute head with otx universal head."""
        ignore = cfg.get("ignore", False)
        for head in ("decode_head", "auxiliary_head"):
            decode_head = cfg.model.get(head, None)
            if decode_head is not None:
                decode_head.base_type = decode_head.type
                decode_head.type = otx_head_factory
                if ignore:
                    cfg_loss_decode = ConfigDict(
                        type="CrossEntropyLossWithIgnore",
                        use_sigmoid=False,
                        loss_weight=1.0,
                    )
                    decode_head.loss_decode = cfg_loss_decode

    # pylint: disable=too-many-branches
    def _configure_head(self, cfg: Config) -> None:
        """Patch number of classes of head."""
        if "decode_head" in cfg.model:
            decode_head = cfg.model.decode_head
            if isinstance(decode_head, list):
                for head in decode_head:
                    head.num_classes = len(self.model_classes)
            else:
                decode_head.num_classes = len(self.model_classes)

            # For SupConDetCon
            if "SupConDetCon" in cfg.model.type:
                cfg.model.num_classes = len(self.model_classes)

        if "auxiliary_head" in cfg.model:
            cfg.model.auxiliary_head.num_classes = len(self.model_classes)

    def configure_ckpt(self, cfg: Config, model_ckpt: str) -> None:
        """Patch checkpoint path for pretrained weight.

        Replace cfg.load_from to model_ckpt
        Replace cfg.load_from to pretrained
        Replace cfg.resume_from to cfg.load_from
        """
        super().configure_ckpt(cfg, model_ckpt)
        # patch checkpoint if needed (e.g. pretrained weights from mmseg)
        if cfg.get("load_from", None) and not model_ckpt and not cfg.get("resume", False):
            cfg.load_from = self.patch_chkpt(cfg.load_from)

    @staticmethod
    def patch_chkpt(ckpt_path: str, new_path: Optional[str] = None) -> str:
        """Modify state dict for pretrained weights to match model state dict."""
        ckpt = CheckpointLoader.load_checkpoint(ckpt_path, map_location="cpu")
        local_torch_hub_folder = torch.hub.get_dir()
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
            new_ckpt = OrderedDict()
            modified = False
            # patch pre-trained checkpoint for model
            for name in ckpt:
                # we should add backbone prefix to backbone parameters names to load it for our models
                if not name.startswith("backbone") and "head" not in name:
                    new_name = "backbone." + name
                    modified = True
                else:
                    new_name = name
                new_ckpt[new_name] = ckpt[name]
            if modified:
                if not new_path:
                    new_path = os.path.join(local_torch_hub_folder, "converted.pth")
                new_path = append_dist_rank_suffix(new_path)
                torch.save(new_ckpt, new_path)
                return new_path
        return ckpt_path

    @staticmethod
    def configure_input_size(
        cfg, input_size_config: InputSizePreset = InputSizePreset.DEFAULT, model_ckpt: Optional[str] = None
    ):
        """Change input size if necessary."""
        input_size = get_configured_input_size(input_size_config, model_ckpt)
        if input_size is None:
            return

        # segmentation models have different input size in train and val data pipeline
        base_input_size = {
            "train": 512,
            "val": 544,
            "test": 544,
            "unlabeled": 512,
        }

        InputSizeManager(cfg.data, base_input_size).set_input_size(input_size)
        logger.info("Input size is changed to {}".format(input_size))


class IncrSegmentationConfigurer(SegmentationConfigurer):
    """Patch config to support incremental learning for semantic segmentation."""

    def configure_task(self, cfg: ConfigDict) -> None:
        """Patch config to support incremental learning."""
        super().configure_task(cfg)

        # TODO: Revisit this part when removing bg label -> it should be 1 because of 'background' label
        if len(set(self.org_model_classes) & set(self.model_classes)) == 1 or set(self.org_model_classes) == set(
            self.model_classes
        ):
            is_cls_incr = False
        else:
            is_cls_incr = True

        # Update TaskAdaptHook (use incremental sampler)
        task_adapt_hook = ConfigDict(
            type="TaskAdaptHook",
            src_classes=self.org_model_classes,
            dst_classes=self.model_classes,
            model_type=cfg.model.type,
            sampler_flag=is_cls_incr,
            efficient_mode=cfg["task_adapt"].get("efficient_mode", False),
        )
        update_or_add_custom_hook(cfg, task_adapt_hook)


class SemiSLSegmentationConfigurer(SegmentationConfigurer):
    """Patch config to support semi supervised learning for semantic segmentation."""

    def configure_hook(self, cfg):
        """Update cfg.custom_hooks."""
        super().configure_hook(cfg)
        # Set unlabeled data hook
        if self.training:
            if cfg.data.get("unlabeled", False) and cfg.data.unlabeled.get("otx_dataset", False):
                self.configure_unlabeled_dataloader(cfg)

    def configure_task(self, cfg: ConfigDict, **kwargs: Any) -> None:
        """Adjust settings for task adaptation."""
        super().configure_task(cfg, **kwargs)

        # Remove task adapt hook (set default torch random sampler)
        remove_custom_hook(cfg, "TaskAdaptHook")

    @staticmethod
    def configure_unlabeled_dataloader(cfg: ConfigDict) -> None:
        """Patch for unlabled dataloader."""

        model_task: Dict[str, str] = {
            "classification": "mmcls",
            "detection": "mmdet",
            "segmentation": "mmseg",
        }  # noqa
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
                    custom_hook["data_loaders"] = [
                        *custom_hook["data_loaders"],
                        unlabeled_dataloader,
                    ]
                    updated = True
            if not updated:
                custom_hooks.append(
                    ConfigDict(
                        type="ComposedDataLoadersHook",
                        data_loaders=unlabeled_dataloader,
                    )
                )
            cfg.custom_hooks = custom_hooks
