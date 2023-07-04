"""Custom universal class incremental otx head."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from typing import Dict, List, Optional

import torch
from mmcv.runner import force_fp32
from mmseg.models.decode_heads.fcn_head import FCNHead
from mmseg.models.decode_heads.sep_aspp_head import DepthwiseSeparableASPPHead
from mmseg.models.losses import accuracy
from mmseg.ops import resize
from torch import nn

from otx.algorithms.segmentation.adapters.mmseg.models.utils import IterativeAggregator
from otx.algorithms.segmentation.adapters.mmseg.utils import (
    get_valid_label_mask_per_batch,
)

from .light_ham import LightHamHead

KNOWN_HEADS = {
    "FCNHead": FCNHead,
    "ASPPHead": DepthwiseSeparableASPPHead,
    "LightHamHead": LightHamHead,
}


def otx_head_factory(*args, base_type="FCNHead", **kwargs):
    """Factory function for creating custom otx head based on mmsegmentation heads."""

    head_base_cls = KNOWN_HEADS[base_type]

    class CustomOTXHead(head_base_cls):
        """Custom OTX head for Semantic Segmentation.

        This Head added head aggregator used in Lite-HRNet by
        DepthwiseSeparableConvModule.
        Please refer to https://github.com/HRNet/Lite-HRNet.
        It also provides interface for incremental learning
        inside OTX framework.

        Args:
            base_type (bool): base type of segmentation head
            enable_aggregator (bool): If true, will use aggregator
                concating all inputs from backbone by DepthwiseSeparableConvModule.
            aggregator_min_channels (int, optional): The number of channels of output of aggregator.
                It would work only if enable_aggregator is true.
            aggregator_merge_norm (str, optional): normalize the output of expanders of aggregator.
                options : "none", "channel", or None
            aggregator_use_concat (str, optional): Whether to concat the last input
                with the output of expanders.
        """

        def __init__(
            self,
            base_type: str = "FCNHead",
            enable_aggregator: bool = False,
            aggregator_min_channels: Optional[int] = None,
            aggregator_merge_norm: Optional[str] = None,
            aggregator_use_concat: bool = False,
            *args,
            **kwargs
        ):

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
            if self.act_cfg and base_type == "FCNHead":
                self.convs[-1].with_activation = False
                delattr(self.convs[-1], "activate")

            if kwargs.get("init_cfg", {}):
                self.init_weights()

        def _transform_inputs(self, inputs: torch.Tensor):
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
                loss_only (bool): If true computes loss only without head forward

            Returns:
                dict[str, Tensor]: a dictionary of loss components
            """
            # is loss_only is True -> inputs are already model logits
            seg_logits = self(inputs) if not loss_only else inputs
            valid_label_mask = get_valid_label_mask_per_batch(img_metas, self.num_classes)
            losses = self.losses(seg_logits, gt_semantic_seg, valid_label_mask=valid_label_mask)
            return losses

        @force_fp32(apply_to=("seg_logit",))
        def losses(
            self,
            seg_logit: torch.Tensor,
            seg_label: torch.Tensor,
            valid_label_mask: Optional[torch.Tensor] = None,
        ):
            """Compute segmentation loss."""
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

    return CustomOTXHead(base_type, *args, **kwargs)
