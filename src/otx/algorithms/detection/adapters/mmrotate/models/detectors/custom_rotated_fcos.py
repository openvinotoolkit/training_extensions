"""OTX RotatedRetinaNet Class for mmrotate detectors."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import functools

from mmrotate.models.builder import ROTATED_DETECTORS
from mmrotate.models.detectors import RotatedFCOS

from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.common.utils.task_adapt import map_class_names

logger = get_logger()


@ROTATED_DETECTORS.register_module()
class CustomRotatedFCOS(RotatedFCOS):
    """SAM optimizer & L2SP regularizer enabled custom RotatedRetinaNet."""

    def __init__(self, *args, task_adapt=None, **kwargs):
        super().__init__(*args, **kwargs)

        # Hook for class-sensitive weight loading
        if task_adapt:
            self._register_load_state_dict_pre_hook(
                functools.partial(
                    self.load_state_dict_pre_hook,
                    self,  # model
                    task_adapt["dst_classes"],  # model_classes
                    task_adapt["src_classes"],  # chkpt_classes
                )
            )

    @staticmethod
    def load_state_dict_pre_hook(model, model_classes, chkpt_classes, chkpt_dict, prefix, *args, **kwargs):
        """Modify input state_dict according to class name matching before weight loading."""
        logger.info(f"----------------- CustomRotatedRetinaNet.load_state_dict_pre_hook() called w/ prefix: {prefix}")

        # Dst to src mapping index
        model_classes = list(model_classes)
        chkpt_classes = list(chkpt_classes)
        model2chkpt = map_class_names(model_classes, chkpt_classes)
        logger.info(f"{chkpt_classes} -> {model_classes} ({model2chkpt})")

        model_dict = model.state_dict()
        # TODO[EUGENE]: need to check the correctness of incremental learning
        param_names = [
            "bbox_head.conv_cls.weight",
            "bbox_head.conv_cls.bias",
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
