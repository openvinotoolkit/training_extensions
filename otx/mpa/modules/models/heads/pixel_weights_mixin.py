# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch.nn as nn
from mmcv.runner import force_fp32
from mmseg.core import add_prefix
from mmseg.models.losses import accuracy
from mmseg.ops import resize

from otx.mpa.modules.utils.seg_utils import get_valid_label_mask_per_batch

from ..losses.utils import LossEqualizer


class PixelWeightsMixin(nn.Module):
    def __init__(
        self,
        enable_loss_equalizer=False,
        loss_target="gt_semantic_seg",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.enable_loss_equalizer = enable_loss_equalizer
        self.loss_target = loss_target

        self.loss_equalizer = None
        if enable_loss_equalizer:
            self.loss_equalizer = LossEqualizer()

        self.forward_output = None

    @property
    def loss_target_name(self):
        return self.loss_target

    @property
    def last_scale(self):
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        num_losses = len(losses_decode)
        if num_losses <= 0:
            return 1.0

        loss_module = losses_decode[0]
        if not hasattr(loss_module, "last_scale"):
            return 1.0

        return loss_module.last_scale

    def set_step_params(self, init_iter, epoch_size):
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        for loss_module in losses_decode:
            if hasattr(loss_module, "set_step_params"):
                loss_module.set_step_params(init_iter, epoch_size)

    def forward_train(
        self,
        inputs,
        img_metas,
        gt_semantic_seg,
        train_cfg,
        pixel_weights=None,
        return_logits=False,
    ):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.
            pixel_weights (Tensor): Pixels weights.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg, train_cfg, pixel_weights)

        if return_logits:
            logits = self.forward_output if self.forward_output is not None else seg_logits
            return losses, logits
        return losses

    @force_fp32(apply_to=("seg_logit",))
    def losses(self, seg_logit, seg_label, train_cfg, pixel_weights=None):
        """Compute segmentation loss."""

        loss = dict()

        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )

        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        out_losses = dict()
        for loss_idx, loss_module in enumerate(losses_decode):
            loss_value, loss_meta = loss_module(seg_logit, seg_label, pixel_weights=pixel_weights)

            loss_name = loss_module.name + f"-{loss_idx}"
            out_losses[loss_name] = loss_value
            loss.update(add_prefix(loss_meta, loss_name))

        if self.enable_loss_equalizer and len(losses_decode) > 1:
            out_losses = self.loss_equalizer.reweight(out_losses)

        for loss_name, loss_value in out_losses.items():
            loss[loss_name] = loss_value

        loss["loss_seg"] = sum(out_losses.values())
        loss["acc_seg"] = accuracy(seg_logit, seg_label)

        return loss


class PixelWeightsMixin2(PixelWeightsMixin):
    def forward_train(
        self,
        inputs,
        img_metas,
        gt_semantic_seg,
        train_cfg,
        pixel_weights=None,
        return_logits=False,
    ):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', 'img_norm_cfg',
                and 'ignored_labels'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.
            pixel_weights (Tensor): Pixels weights.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self(inputs)
        valid_label_mask = get_valid_label_mask_per_batch(img_metas, self.num_classes)
        losses = self.losses(
            seg_logits, gt_semantic_seg, train_cfg, valid_label_mask=valid_label_mask, pixel_weights=pixel_weights
        )

        if return_logits:
            logits = self.forward_output if self.forward_output is not None else seg_logits
            return losses, logits
        return losses

    @force_fp32(apply_to=("seg_logit",))
    def losses(self, seg_logit, seg_label, train_cfg, valid_label_mask, pixel_weights=None):
        """Compute segmentation loss."""

        loss = dict()

        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )

        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        out_losses = dict()
        for loss_idx, loss_module in enumerate(losses_decode):
            loss_value, loss_meta = loss_module(seg_logit, seg_label, valid_label_mask, pixel_weights=pixel_weights)

            loss_name = loss_module.name + f"-{loss_idx}"
            out_losses[loss_name] = loss_value
            loss.update(add_prefix(loss_meta, loss_name))

        if self.enable_loss_equalizer and len(losses_decode) > 1:
            out_losses = self.loss_equalizer.reweight(out_losses)

        for loss_name, loss_value in out_losses.items():
            loss[loss_name] = loss_value

        loss["loss_seg"] = sum(out_losses.values())
        loss["acc_seg"] = accuracy(seg_logit, seg_label)

        return loss
