# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.runner import force_fp32
from mmseg.core import add_prefix
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.ocr_head import OCRHead
from mmseg.models.losses import accuracy
from mmseg.ops import resize

from otx.mpa.modules.utils.seg_utils import get_valid_label_mask_per_batch


@HEADS.register_module()
class CustomOCRHead(OCRHead):
    """Custom Object-Contextual Representations for Semantic Segmentation."""

    def forward_train(self, inputs, prev_output, img_metas, gt_semantic_seg, train_cfg, pixel_weights=None):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            prev_output (Tensor): The output of previous decode head.
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
        seg_logits = self.forward(inputs, prev_output)
        valid_label_mask = get_valid_label_mask_per_batch(img_metas, self.num_classes)
        losses = self.losses(seg_logits, gt_semantic_seg, valid_label_mask, train_cfg, pixel_weights)

        return losses, seg_logits

    @force_fp32(apply_to=("seg_logit",))
    def losses(self, seg_logit, seg_label, valid_label_mask, train_cfg, pixel_weights=None):
        """Compute segmentation loss."""

        loss = dict()

        seg_logit = resize(input=seg_logit, size=seg_label.shape[2:], mode="bilinear", align_corners=self.align_corners)

        seg_label = seg_label.squeeze(1)
        out_losses = dict()
        for loss_idx, loss_module in enumerate(self.loss_modules):
            loss_value, loss_meta = loss_module(seg_logit, seg_label, valid_label_mask, pixel_weights=pixel_weights)

            loss_name = loss_module.name + f"-{loss_idx}"
            out_losses[loss_name] = loss_value
            loss.update(add_prefix(loss_meta, loss_name))

        if self.enable_loss_equalizer and len(self.loss_modules) > 1:
            out_losses = self.loss_equalizer.reweight(out_losses)

        for loss_name, loss_value in out_losses.items():
            loss[loss_name] = loss_value

        loss["loss_seg"] = sum(out_losses.values())
        loss["acc_seg"] = accuracy(seg_logit, seg_label)

        if train_cfg.mix_loss.enable:
            mix_loss = self._mix_loss(seg_logit, seg_label, ignore_index=self.ignore_index)

            mix_loss_weight = train_cfg.mix_loss.get("weight", 1.0)
            loss["loss_mix"] = mix_loss_weight * mix_loss

        return loss
