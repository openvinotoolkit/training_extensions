"""Encoder-decoder for incremental learning."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import functools

import torch
from mmseg.models import SEGMENTORS
from mmseg.utils import get_root_logger

from otx.algorithms.common.utils.task_adapt import map_class_names

from .mixin import PixelWeightsMixin
from .otx_encoder_decoder import OTXEncoderDecoder


@SEGMENTORS.register_module()
class ClassIncrEncoderDecoder(PixelWeightsMixin, OTXEncoderDecoder):
    """Encoder-decoder for incremental learning."""

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

    def forward_train(
        self,
        img,
        img_metas,
        gt_semantic_seg,
        aux_img=None,
        **kwargs,
    ):  # pylint: disable=arguments-renamed
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            aux_img (Tensor): Auxiliary images.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if aux_img is not None:
            mix_loss_enabled = False
            mix_loss_cfg = self.train_cfg.get("mix_loss", None)
            if mix_loss_cfg is not None:
                mix_loss_enabled = mix_loss_cfg.get("enable", False)
            if mix_loss_enabled:
                self.train_cfg.mix_loss.enable = mix_loss_enabled

        if self.train_cfg.mix_loss.enable:
            img = torch.cat([img, aux_img], dim=0)
            gt_semantic_seg = torch.cat([gt_semantic_seg, gt_semantic_seg], dim=0)

        return super().forward_train(img, img_metas, gt_semantic_seg, **kwargs)

    @staticmethod
    def load_state_dict_pre_hook(
        model, model_classes, chkpt_classes, chkpt_dict, prefix, *args, **kwargs
    ):  # pylint: disable=too-many-locals, unused-argument
        """Modify input state_dict according to class name matching before weight loading."""
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
            for model_key, c in enumerate(model2chkpt):
                if c >= 0:
                    model_param[model_key].copy_(chkpt_param[c])

            # Replace checkpoint weight by mixed weights
            chkpt_dict[chkpt_name] = model_param
