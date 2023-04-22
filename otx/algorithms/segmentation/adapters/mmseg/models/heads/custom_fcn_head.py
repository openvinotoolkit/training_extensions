"""Custom FCN head."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.runner import force_fp32
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.fcn_head import FCNHead
from mmseg.ops import resize
from mmseg.models.losses import accuracy
from otx.algorithms.segmentation.adapters.mmseg.models.utils import IterativeAggregator, LossEqualizer
import torch.nn as nn
from otx.algorithms.segmentation.adapters.mmseg.utils import (
    get_valid_label_mask_per_batch,
)

@HEADS.register_module()
class CustomFCNHead(FCNHead):  # pylint: disable=too-many-ancestors
    """Custom Fully Convolution Networks for Semantic Segmentation."""
    def __init__(self,
        enable_aggregator=False,
        aggregator_min_channels=None,
        aggregator_merge_norm=None,
        aggregator_use_concat=False,
        enable_loss_equalizer=False,
        *args, **kwargs):

        in_channels = kwargs.get("in_channels")
        in_index = kwargs.get("in_index")
        norm_cfg = kwargs.get("norm_cfg")
        conv_cfg = kwargs.get("conv_cfg")
        input_transform = kwargs.get("input_transform")
        self.loss_equalizer = None

        aggregator = None
        if enable_aggregator: # Lite-HRNet aggregator
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

        if enable_loss_equalizer:
            self.loss_equalizer = LossEqualizer()

        self.aggregator = aggregator

        # re-define variables
        self.in_channels = in_channels
        self.input_transform = input_transform
        self.in_index = in_index
        
        self.ignore_index = 255

        # get rid of last activation of convs module
        if self.act_cfg:
            self.convs[-1].with_activation = False
            delattr(self.convs[-1], "activate")

        if kwargs.get("init_cfg", {}):
            self.init_weights()


    def _transform_inputs(self, inputs):
        if self.aggregator is not None:
            inputs = self.aggregator(inputs)[0]
        else:
            inputs = super()._transform_inputs(inputs)

        return inputs


    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
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

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self(inputs)
        valid_label_mask = get_valid_label_mask_per_batch(img_metas, self.num_classes)
        losses = self.losses(seg_logits, gt_semantic_seg, valid_label_mask=valid_label_mask)
        return losses

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label, valid_label_mask=None):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index, 
                    valid_label_mask=valid_label_mask)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index, 
                    valid_label_mask=valid_label_mask)

        loss['acc_seg'] = accuracy(
            seg_logit, seg_label, ignore_index=self.ignore_index)

        if self.loss_equalizer:
            out_loss = self.loss_equalizer.reweight(loss)
            for loss_name, loss_value in out_loss.items():
                loss[loss_name] = loss_value

        return loss