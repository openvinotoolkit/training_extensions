"""Base configurer for mmseg config."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from collections import OrderedDict
from typing import Any, Optional, Tuple

import torch
from mmcv.runner import CheckpointLoader
from mmcv.utils import Config, ConfigDict

from otx.algorithms.common.adapters.mmcv.clsincr_mixin import IncrConfigurerMixin
from otx.algorithms.common.adapters.mmcv.configurer import BaseConfigurer
from otx.algorithms.common.adapters.mmcv.semisl_mixin import SemiSLConfigurerMixin
from otx.algorithms.common.adapters.mmcv.utils.config_utils import (
    InputSizeManager,
    remove_custom_hook,
)
from otx.algorithms.common.utils import append_dist_rank_suffix
from otx.algorithms.segmentation.adapters.mmseg.models.heads import otx_head_factory
from otx.utils.logger import get_logger

logger = get_logger()


# pylint: disable=too-many-public-methods
class SegmentationConfigurer(BaseConfigurer):
    """Patch config to support otx train."""

    def configure_task(
        self,
        cfg: Config,
        **kwargs,
    ) -> None:
        """Patch config to support training algorithm."""
        super().configure_task(cfg, **kwargs)
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

    def configure_ckpt(self, cfg: Config, model_ckpt_path: str) -> None:
        """Patch checkpoint path for pretrained weight.

        Replace cfg.load_from to model_ckpt_path
        Replace cfg.load_from to pretrained
        Replace cfg.resume_from to cfg.load_from
        """
        super().configure_ckpt(cfg, model_ckpt_path)
        # patch checkpoint if needed (e.g. pretrained weights from mmseg)
        if cfg.get("load_from", None) and not model_ckpt_path and not cfg.get("resume", False):
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
        cfg, input_size=Optional[Tuple[int, int]], model_ckpt_path: Optional[str] = None, training=True
    ):
        """Change input size if necessary."""
        if input_size is None:  # InputSizePreset.DEFAULT
            return

        # Segmentation models have different input size in train and val data pipeline
        base_input_size = {
            "train": 512,
            "val": 544,
            "test": 544,
            "unlabeled": 512,
        }
        manager = InputSizeManager(cfg, base_input_size)

        if input_size == (0, 0):  # InputSizePreset.AUTO
            if training:
                input_size = BaseConfigurer.adapt_input_size_to_dataset(cfg, manager, use_annotations=False)
            else:
                input_size = manager.get_trained_input_size(model_ckpt_path)
            if input_size is None:
                return

        manager.set_input_size(input_size)
        logger.info("Input size is changed to {}".format(input_size))


class IncrSegmentationConfigurer(IncrConfigurerMixin, SegmentationConfigurer):
    """Patch config to support incremental learning for semantic segmentation."""

    def is_incremental(self) -> bool:
        """Return whether current model classes is increased from original model classes."""
        # TODO: Revisit this part when removing bg label -> it should be 1 because of 'background' label
        if len(set(self.org_model_classes) & set(self.model_classes)) == 1 or set(self.org_model_classes) == set(
            self.model_classes
        ):
            is_cls_incr = False
        else:
            is_cls_incr = True
        return is_cls_incr


class SemiSLSegmentationConfigurer(SemiSLConfigurerMixin, SegmentationConfigurer):
    """Patch config to support semi supervised learning for semantic segmentation."""

    def configure_task(self, cfg: ConfigDict, **kwargs: Any) -> None:
        """Adjust settings for task adaptation."""
        super().configure_task(cfg, **kwargs)

        # Remove task adapt hook (set default torch random sampler)
        remove_custom_hook(cfg, "TaskAdaptHook")
