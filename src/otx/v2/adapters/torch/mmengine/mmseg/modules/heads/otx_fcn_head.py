"""Custom universal class incremental otx head."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from typing import Dict, List, Optional

import torch
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.fcn_head import FCNHead
from mmseg.models.losses import accuracy
from mmseg.models.utils import resize
from torch import nn

from otx.v2.adapters.torch.mmengine.mmseg.modules.models.utils import IterativeAggregator
from otx.v2.adapters.torch.mmengine.mmseg.utils.data_utils import get_valid_label_mask_per_batch


@HEADS.register_module()
class OTXFCNHead(FCNHead):
    """OTXFCNHead is a fully convolutional network head used in OTX.

    Args:
        enable_aggregator (bool): Whether to enable the Lite-HRNet aggregator.
        aggregator_min_channels (int, optional): Minimum number of channels for the aggregator.
        aggregator_merge_norm (str, optional): Type of normalization to use for the aggregator.
        aggregator_use_concat (bool): Whether to use concatenation for the aggregator.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        aggregator (IterativeAggregator): The Lite-HRNet aggregator.
        in_channels (int): Number of input channels.
        input_transform (dict): Input transformation.
        in_index (int): Index of input.
        ignore_index (int): Index to ignore.

    """

    def __init__(
        self,
        enable_aggregator: bool = False,
        aggregator_min_channels: Optional[int] = None,
        aggregator_merge_norm: Optional[str] = None,
        aggregator_use_concat: bool = False,
        *args,
        **kwargs
    ):
        """
        Initializes OTXFCNHead.

        Args:
            enable_aggregator (bool): Whether to enable the Lite-HRNet aggregator.
            aggregator_min_channels (int, optional): Minimum number of channels for the aggregator.
            aggregator_merge_norm (str, optional): Type of normalization to use for the aggregator.
            aggregator_use_concat (bool): Whether to use concatenation for the aggregator.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        in_channels = kwargs.get("in_channels")
        in_index = kwargs.get("in_index")
        norm_cfg = kwargs.get("norm_cfg")
        conv_cfg = kwargs.get("conv_cfg")
        input_transform = kwargs.get("input_transform")

        aggregator = None
        if enable_aggregator:  # Lite-HRNet aggregator
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

        self.ignore_index = 255

        # get rid of last activation of convs module
        if self.act_cfg:
            self.convs[-1].with_activation = False
            delattr(self.convs[-1], "activate")

        if kwargs.get("init_cfg", {}):
            self.init_weights()

    def _transform_inputs(self, inputs: torch.Tensor):
        """
        Transforms inputs.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed input tensor.

        """
        if self.aggregator is not None:
            inputs = self.aggregator(inputs)[0]
        else:
            inputs = super()._transform_inputs(inputs)

        return inputs

    def forward_train(
        self,
        inputs: torch.Tensor,
        img_metas: List[Dict],
        gt_semantic_seg: torch.Tensor,
        train_cfg: Dict = dict(),
        loss_only: bool = False,
    ):
        """
        Forward function for training.

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
            loss_only (bool): If true computes loss only without head forward

        Returns:
            dict[str, Tensor]: a dictionary of loss components

        """
        # is loss_only is True -> inputs are already model logits
        seg_logits = self(inputs) if not loss_only else inputs
        valid_label_mask = get_valid_label_mask_per_batch(img_metas, self.num_classes)
        losses = self.losses(seg_logits, gt_semantic_seg, valid_label_mask=valid_label_mask)
        return losses

    def losses(
        self,
        seg_logit: torch.Tensor,
        seg_label: torch.Tensor,
        valid_label_mask: Optional[torch.Tensor] = None,
    ):
        """
        Compute segmentation loss.

        Args:
            seg_logit (torch.Tensor): Logits tensor.
            seg_label (torch.Tensor): Label tensor.
            valid_label_mask (torch.Tensor, optional): Valid label mask tensor.

        Returns:
            dict: A dictionary of loss components.

        """
        loss = dict()

        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
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
            valid_label_mask_cfg = dict()
            if loss_decode.loss_name == "loss_ce_ignore":
                valid_label_mask_cfg["valid_label_mask"] = valid_label_mask
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logit, seg_label, weight=seg_weight, ignore_index=self.ignore_index, **valid_label_mask_cfg
                )
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logit, seg_label, weight=seg_weight, ignore_index=self.ignore_index, **valid_label_mask_cfg
                )

        loss["acc_seg"] = accuracy(seg_logit, seg_label, ignore_index=self.ignore_index)

        return loss
