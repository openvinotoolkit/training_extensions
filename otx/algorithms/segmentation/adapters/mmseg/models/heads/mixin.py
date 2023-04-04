"""Modules for aggregator and loss mix."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmseg.core import add_prefix
from mmseg.models.losses import accuracy
from mmseg.ops import resize
from torch import nn

from otx.algorithms.segmentation.adapters.mmseg.models.utils import (
    AngularPWConv,
    IterativeAggregator,
    LossEqualizer,
    normalize,
)
from otx.algorithms.segmentation.adapters.mmseg.utils import (
    get_valid_label_mask_per_batch,
)

# pylint: disable=abstract-method, unused-argument, keyword-arg-before-vararg


class SegmentOutNormMixin(nn.Module):
    """SegmentOutNormMixin."""

    def __init__(self, *args, enable_out_seg=True, enable_out_norm=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.enable_out_seg = enable_out_seg
        self.enable_out_norm = enable_out_norm

        if enable_out_seg:
            if enable_out_norm:
                self.conv_seg = AngularPWConv(self.channels, self.out_channels, clip_output=True)
        else:
            self.conv_seg = None

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        if self.enable_out_norm:
            feat = normalize(feat, dim=1, p=2)
        if self.conv_seg is not None:
            return self.conv_seg(feat)
        return feat


class AggregatorMixin(nn.Module):
    """A class for creating an aggregator."""

    def __init__(
        self,
        *args,
        enable_aggregator=False,
        aggregator_min_channels=None,
        aggregator_merge_norm=None,
        aggregator_use_concat=False,
        **kwargs,
    ):

        in_channels = kwargs.get("in_channels")
        in_index = kwargs.get("in_index")
        norm_cfg = kwargs.get("norm_cfg")
        conv_cfg = kwargs.get("conv_cfg")
        input_transform = kwargs.get("input_transform")

        aggregator = None
        if enable_aggregator:
            assert isinstance(in_channels, (tuple, list))
            assert len(in_channels) > 1

            aggregator = IterativeAggregator(
                in_channels=in_channels,
                min_channels=aggregator_min_channels,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                merge_norm=aggregator_merge_norm,
                use_concat=aggregator_use_concat,
            )

            aggregator_min_channels = aggregator_min_channels if aggregator_min_channels is not None else 0
            # change arguments temporarily
            kwargs["in_channels"] = max(in_channels[0], aggregator_min_channels)
            kwargs["input_transform"] = None
            if in_index is not None:
                kwargs["in_index"] = in_index[0]

        super().__init__(*args, **kwargs)

        self.aggregator = aggregator
        # re-define variables
        self.in_channels = in_channels
        self.input_transform = input_transform
        self.in_index = in_index

    def _transform_inputs(self, inputs):
        inputs = super()._transform_inputs(inputs)
        if self.aggregator is not None:
            inputs = self.aggregator(inputs)[0]
        return inputs


class MixLossMixin(nn.Module):
    """Loss mixing module."""

    @staticmethod
    def _mix_loss(logits, target, ignore_index=255):
        num_samples = logits.size(0)
        assert num_samples % 2 == 0

        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            probs_a, probs_b = torch.split(probs, num_samples // 2)
            mean_probs = 0.5 * (probs_a + probs_b)
            trg_probs = torch.cat([mean_probs, mean_probs], dim=0)

        log_probs = torch.log_softmax(logits, dim=1)
        losses = torch.sum(trg_probs * log_probs, dim=1).neg()

        valid_mask = target != ignore_index
        valid_losses = torch.where(valid_mask, losses, torch.zeros_like(losses))

        return valid_losses.mean()

    @force_fp32(apply_to=("seg_logit",))
    def losses(self, seg_logit, seg_label, train_cfg, *args, **kwargs):
        """Loss computing."""
        loss = super().losses(seg_logit, seg_label, train_cfg, *args, **kwargs)
        if train_cfg.get("mix_loss", None) and train_cfg.mix_loss.get("enable", False):
            mix_loss = self._mix_loss(seg_logit, seg_label, ignore_index=self.ignore_index)

            mix_loss_weight = train_cfg.mix_loss.get("weight", 1.0)
            loss["loss_mix"] = mix_loss_weight * mix_loss

        return loss


class PixelWeightsMixin(nn.Module):
    """PixelWeightsMixin."""

    def __init__(self, enable_loss_equalizer=False, loss_target="gt_semantic_seg", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.enable_loss_equalizer = enable_loss_equalizer
        self.loss_target = loss_target

        self.loss_equalizer = None
        if enable_loss_equalizer:
            self.loss_equalizer = LossEqualizer()

        self.forward_output = None

    @property
    def loss_target_name(self):
        """Return loss target name."""
        return self.loss_target

    @property
    def last_scale(self):
        """Return the last scale."""
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
        """Set step parameters."""
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
    """Pixel weight mixin class."""

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
    def losses(
        self, seg_logit, seg_label, train_cfg, valid_label_mask, pixel_weights=None
    ):  # pylint: disable=arguments-renamed
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
