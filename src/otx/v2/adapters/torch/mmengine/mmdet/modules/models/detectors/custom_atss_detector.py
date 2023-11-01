"""OTX ATSS Class for mmdetection detectors."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn

from mmdet.models.detectors.atss import ATSS
from mmdet.registry import MODELS

from otx.v2.adapters.torch.modules.utils.task_adapt import map_class_names
from otx.v2.api.utils.logger import get_logger

logger = get_logger()


@MODELS.register_module()
class CustomATSS(ATSS):
    """SAM optimizer & L2SP regularizer enabled custom ATSS."""

    def __init__(self, *args, task_adapt: dict | None = None, **kwargs) -> None:
        """Initialize CustomATSS detector."""
        super().__init__(*args, **kwargs)

        # Hook for class-sensitive weight loading
        if task_adapt:
            self._register_load_state_dict_pre_hook(
                functools.partial(
                    self.load_state_dict_pre_hook,
                    self,  # model
                    task_adapt["dst_classes"],  # model_classes
                    task_adapt["src_classes"],  # chkpt_classes
                ),
            )

    @staticmethod
    def load_state_dict_pre_hook(
        model: nn.Module,
        model_classes: list[str],
        chkpt_classes: list[str],
        chkpt_dict: dict,
        prefix: str,
    ) -> None:
        """Modify input state_dict according to class name matching before weight loading."""
        logger.info(f"----------------- CustomATSS.load_state_dict_pre_hook() called w/ prefix: {prefix}")

        # Dst to src mapping index
        model_classes = list(model_classes)
        chkpt_classes = list(chkpt_classes)
        model2chkpt = map_class_names(model_classes, chkpt_classes)
        logger.info(f"{chkpt_classes} -> {model_classes} ({model2chkpt})")

        model_dict = model.state_dict()
        param_names = [
            "bbox_head.atss_cls.weight",
            "bbox_head.atss_cls.bias",
        ]
        for model_name in param_names:
            chkpt_name = prefix + model_name
            if model_name not in model_dict or chkpt_name not in chkpt_dict:
                logger.info(f"Skipping weight copy: {chkpt_name}")
                continue

            # Mix weights
            model_param = model_dict[model_name].clone()
            chkpt_param = chkpt_dict[chkpt_name]
            for model_t, ckpt_t in enumerate(model2chkpt):
                if ckpt_t >= 0:
                    model_param[model_t].copy_(chkpt_param[ckpt_t])

            # Replace checkpoint weight by mixed weights
            chkpt_dict[chkpt_name] = model_param
