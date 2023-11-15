"""OTX MaskRCNN Class for mmdetection detectors."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from mmdet.models.detectors.mask_rcnn import MaskRCNN
from mmdet.registry import MODELS

if TYPE_CHECKING:
    from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
    from torch import Tensor, nn

from otx.v2.adapters.torch.modules.utils.task_adapt import map_class_names
from otx.v2.api.utils.logger import get_logger

logger = get_logger()

# pylint: disable=too-many-locals, protected-access, unused-argument


@MODELS.register_module()
class CustomMaskRCNN(MaskRCNN):
    """CustomMaskRCNN Class for mmdetection detectors."""

    def __init__(
        self,
        backbone: ConfigType,
        rpn_head: ConfigType,
        roi_head: ConfigType,
        train_cfg: OptConfigType,
        test_cfg: OptConfigType,
        neck: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
        task_adapt: dict | None = None,
    ) -> None:
        """Initialize CustomMaskRCNN.

        Args:
            backbone (ConfigType): Config for detector's backbone
            rpn_head (ConfigType): Config for detector's rpn_head
            roi_head (ConfigType): Config for detector's roi_head
            train_cfg (OptConfigType): Hyperparams for training
            test_cfg (OptConfigType): Hyperparams for testing
            neck (ConfigType): Config for detector's neck
            data_preprocessor (ConfigType): Config for detector's data_preprocessor
            init_cfg (ConfigType): Hyperparams for initialization
            task_adapt (dict | None): Dictionary contains model classses and checkpoint classes information.
        """
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor,
        )

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
        chkpt_dict: dict[str, Tensor],
        prefix: str,
    ) -> None:
        """Modify input state_dict according to class name matching before weight loading.

        Args:
            model(nn.Module): MaskRCNN detector.
            model_classes(list[str]): Classes of model which come from dataset.
            chkpt_classes(list[str]): Classes of checkpoint of pretrained model.
            chkpt_dict(dict[str, Tensor]): Checkpoint dictionary
            prefix(str): Prefix for key of chkpt_dict
        """
        logger.info(f"----------------- CustomMaskRCNN.load_state_dict_pre_hook() called w/ prefix: {prefix}")

        # Dst to src mapping index
        model_dict = model.state_dict()
        model_classes = list(model_classes)
        chkpt_classes = list(chkpt_classes)
        model2chkpt = map_class_names(model_classes, chkpt_classes)
        logger.info(f"{chkpt_classes} -> {model_classes} ({model2chkpt})")

        # List of class-relevant params & their row-stride
        param_strides = {
            "roi_head.bbox_head.fc_cls.weight": 1,
            "roi_head.bbox_head.fc_cls.bias": 1,
            "roi_head.bbox_head.fc_reg.weight": 4,  # 4 rows (bbox) for each class
            "roi_head.bbox_head.fc_reg.bias": 4,
        }

        for model_name, stride in param_strides.items():
            chkpt_name = prefix + model_name
            if model_name not in model_dict or chkpt_name not in chkpt_dict:
                logger.info(f"Skipping weight copy: {chkpt_name}")
                continue

            # Mix weights
            model_param = model_dict[model_name].clone()
            chkpt_param = chkpt_dict[chkpt_name]
            for model_t, ckpt_t in enumerate(model2chkpt):
                if ckpt_t >= 0:
                    # Copying only matched weight rows
                    model_param[(model_t) * stride : (model_t + 1) * stride].copy_(
                        chkpt_param[(ckpt_t) * stride : (ckpt_t + 1) * stride],
                    )
            if model_param.shape[0] > len(model_classes * stride):  # BG class
                num_ckpt_class = len(chkpt_classes)
                num_model_class = len(model_classes)
                model_param[(num_model_class) * stride : (num_model_class + 1) * stride].copy_(
                    chkpt_param[(num_ckpt_class) * stride : (num_ckpt_class + 1) * stride],
                )

            # Replace checkpoint weight by mixed weights
            chkpt_dict[chkpt_name] = model_param
