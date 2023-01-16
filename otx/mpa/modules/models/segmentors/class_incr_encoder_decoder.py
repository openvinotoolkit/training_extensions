# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import functools

from mmseg.models import SEGMENTORS
from mmseg.utils import get_root_logger

from otx.mpa.modules.utils.task_adapt import map_class_names

from .mix_loss_mixin import MixLossMixin
from .otx_encoder_decoder import OTXEncoderDecoder
from .pixel_weights_mixin import PixelWeightsMixin


@SEGMENTORS.register_module()
class ClassIncrEncoderDecoder(MixLossMixin, PixelWeightsMixin, OTXEncoderDecoder):
    """ """

    def __init__(self, *args, task_adapt=None, **kwargs):
        super().__init__(*args, **kwargs)

        # Hook for class-sensitive weight loading
        assert task_adapt is not None, "When using task_adapt, task_adapt must be set."

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
        """Modify input state_dict according to class name matching before weight loading"""
        logger = get_root_logger("INFO")
        logger.info(f"----------------- ClassIncrEncoderDecoder.load_state_dict_pre_hook() called w/ prefix: {prefix}")

        # Dst to src mapping index
        model_classes = list(model_classes)
        chkpt_classes = list(chkpt_classes)
        model2chkpt = map_class_names(model_classes, chkpt_classes)
        logger.info(f"{chkpt_classes} -> {model_classes} ({model2chkpt})")

        model_dict = model.state_dict()
        param_names = [
            "decode_head.conv_seg.weight",
            "decode_head.conv_seg.bias",
        ]
        for model_name in param_names:
            chkpt_name = prefix + model_name
            if model_name not in model_dict or chkpt_name not in chkpt_dict:
                logger.info(f"Skipping weight copy: {chkpt_name}")
                continue

            # Mix weights
            model_param = model_dict[model_name].clone()
            chkpt_param = chkpt_dict[chkpt_name]
            for m, c in enumerate(model2chkpt):
                if c >= 0:
                    model_param[m].copy_(chkpt_param[c])

            # Replace checkpoint weight by mixed weights
            chkpt_dict[chkpt_name] = model_param
