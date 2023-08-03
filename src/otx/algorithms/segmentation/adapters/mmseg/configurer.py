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
    remove_custom_hook,
    update_or_add_custom_hook,
)
from otx.algorithms.common.utils import append_dist_rank_suffix
from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.segmentation.adapters.mmseg.models.heads import otx_head_factory
from otx.algorithms.segmentation.adapters.mmseg.utils import (
    patch_datasets,
    patch_evaluation,
)

logger = get_logger()


# pylint: disable=too-many-public-methods
class SegmentationConfigurer(BaseConfigurer):
    """Patch config to support otx train."""

    def __init__(self) -> None:
        self.task_adapt_type: Optional[str] = None
        self.task_adapt_op: str = "REPLACE"
        self.org_model_classes: List[str] = []
        self.model_classes: List[str] = []
        self.data_classes: List[str] = []

    # pylint: disable=too-many-arguments
    def configure(
        self,
        cfg: Config,
        model_ckpt: str,
        data_cfg: Config,
        training: bool = True,
        subset: str = "train",
        ir_options: Optional[Config] = None,
        data_classes: Optional[List[str]] = None,
        model_classes: Optional[List[str]] = None,
    ) -> Config:
        """Create MMCV-consumable config from given inputs."""
        logger.info(f"configure!: training={training}")

        self.configure_base(cfg, data_cfg, data_classes, model_classes)
        self.configure_device(cfg, training)
        self.configure_ckpt(cfg, model_ckpt)
        self.configure_model(cfg, ir_options, "segmentation")
        self.configure_data(cfg, training, data_cfg)
        self.configure_task(cfg, training)
        self.configure_hook(cfg)
        self.configure_samples_per_gpu(cfg, subset)
        self.configure_fp16_optimizer(cfg)
        self.configure_compat_cfg(cfg)
        return cfg

    def configure_compatibility(self, cfg, **kwargs):
        """Configure for OTX compatibility with mmseg."""
        options_for_patch_datasets = {"type": "MPASegDataset"}
        patch_datasets(cfg, **options_for_patch_datasets)
        patch_evaluation(cfg)

    def configure_data(
        self,
        cfg: Config,
        training: bool,
        data_cfg: Optional[Config],
    ) -> None:
        """Patch cfg.data.

        Merge cfg and data_cfg
        Wrap original dataset type to MPAsegDataset
        """
        if data_cfg:
            cfg.merge_from_dict(data_cfg)

        train_data_cfg = self.get_data_cfg(cfg, "train")
        for mode in ["train", "val", "test"]:
            if train_data_cfg.type == "MPASegDataset" and cfg.data.get(mode, False):
                if cfg.data[mode]["type"] != "MPASegDataset":
                    # Wrap original dataset config
                    org_type = cfg.data[mode]["type"]
                    cfg.data[mode]["type"] = "MPASegDataset"
                    cfg.data[mode]["org_type"] = org_type

    def configure_task(
        self,
        cfg: Config,
        training: bool,
    ) -> None:
        """Patch config to support training algorithm."""
        super().configure_task(cfg, training)
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


class IncrSegmentationConfigurer(SegmentationConfigurer):
    """Patch config to support incremental learning for semantic segmentation."""

    def configure_task(self, cfg: ConfigDict, training: bool) -> None:
        """Patch config to support incremental learning."""
        super().configure_task(cfg, training)

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

    def configure_data(self, cfg: ConfigDict, training: bool, data_cfg: ConfigDict) -> None:
        """Patch cfg.data."""
        super().configure_data(cfg, training, data_cfg)
        # Set unlabeled data hook
        if training:
            if cfg.data.get("unlabeled", False) and cfg.data.unlabeled.get("otx_dataset", False):
                self.configure_unlabeled_dataloader(cfg)

    def configure_task(self, cfg: ConfigDict, training: bool, **kwargs: Any) -> None:
        """Adjust settings for task adaptation."""
        super().configure_task(cfg, training, **kwargs)

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
