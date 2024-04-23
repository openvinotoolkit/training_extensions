# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Custom FCNHead modules for OTX segmentation model."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from otx.algo.modules import ConvModule
from otx.algo.segmentation.modules import IterativeAggregator

from .base_head import BaseSegmHead


class FCNHead(BaseSegmHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(
        self,
        num_convs: int = 2,
        kernel_size: int = 3,
        concat_input: bool = True,
        dilation: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize a Fully Convolution Networks head.

        Args:
            num_convs (int): Number of convs in the head.
            kernel_size (int): The kernel size for convs in the head.
            concat_input (bool): Whether to concat input and output of convs.
            dilation (int): The dilation rate for convs in the head.
            **kwargs: Additional arguments.
        """
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super().__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            ),
        )
        for _ in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                ),
            )
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            )

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        feats = self.convs(x)
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output

    # class ClassIncrementalMixin:
    #     """Mixin for class incremental learning."""

    #     def loss_by_feat(
    #         self,
    #         seg_logits: Tensor,
    #         batch_data_samples: SampleList,
    #     ) -> dict:
    #         """Compute segmentation loss.

    #         Args:
    #             seg_logits (Tensor): The output from decode head forward function.
    #             batch_data_samples (List[:obj:`SegDataSample`]): The seg
    #                 data samples. It usually includes information such
    #                 as `metainfo` and `gt_sem_seg`.

    #         Returns:
    #             dict[str, Tensor]: a dictionary of loss components
    #         """
    #         img_metas = [data_sample.metainfo for data_sample in batch_data_samples]
    #         valid_label_mask = self.get_valid_label_mask(img_metas)
    #         seg_label = self._stack_batch_gt(batch_data_samples)
    #         loss = {}
    #         seg_logits = resize(
    #             input=seg_logits,
    #             size=seg_label.shape[2:],
    #             mode="bilinear",
    #             align_corners=self.align_corners,
    #         )
    #         seg_weight = self.sampler.sample(seg_logits, seg_label) if self.sampler is not None else None
    #         seg_label = seg_label.squeeze(1)

    #         losses_decode = [self.loss_decode] if not isinstance(self.loss_decode, nn.ModuleList) else self.loss_decode
    #         for loss_decode in losses_decode:
    #             valid_label_mask_cfg = {}
    #             if loss_decode.loss_name == "loss_ce_ignore":
    #                 valid_label_mask_cfg["valid_label_mask"] = valid_label_mask
    #             if loss_decode.loss_name not in loss:
    #                 loss[loss_decode.loss_name] = loss_decode(
    #                     seg_logits,
    #                     seg_label,
    #                     weight=seg_weight,
    #                     ignore_index=self.ignore_index,
    #                     **valid_label_mask_cfg,
    #                 )
    #             else:
    #                 loss[loss_decode.loss_name] += loss_decode(
    #                     seg_logits,
    #                     seg_label,
    #                     weight=seg_weight,
    #                     ignore_index=self.ignore_index,
    #                     valid_label_mask=valid_label_mask,
    #                     **valid_label_mask_cfg,
    #                 )

    #         return loss

    # def get_valid_label_mask(self, img_metas: list[dict]) -> list[torch.Tensor]:
    #     """Get valid label mask removing ignored classes to zero mask in a batch.

    #     Args:
    #         img_metas (List[dict]): List of image metadata.

    #     Returns:
    #         List[torch.Tensor]: List of valid label masks.
    #     """
    #     valid_label_mask = []  # type: List[torch.Tensor]
    #     for meta in img_metas:  # type: dict
    #         mask = torch.Tensor([1 for _ in range(self.num_classes)])  # type: torch.Tensor
    #         if "ignored_labels" in meta and meta["ignored_labels"]:  # type: ignore
    #             mask[meta["ignored_labels"]] = 0  # type: ignore
    #         valid_label_mask.append(mask)
    #     return valid_label_mask


class CustomFCNHead(FCNHead):
    """Custom FCNHead implementation.

    Custom FCNHead supports iterative agregator and ignored label for class
    incremental learning cases.
    """

    def __init__(
        self,
        enable_aggregator: bool = False,
        aggregator_min_channels: int = 0,
        aggregator_merge_norm: str | None = None,
        aggregator_use_concat: bool = False,
        in_channels: list[int] | int | None = None,
        in_index: list[int] | int | None = None,
        norm_cfg: dict[str, Any] | None = None,
        conv_cfg: dict[str, Any] | None = None,
        input_transform: list | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Custom FCNHead initialization.

        Args:
            enable_aggregator (bool, optional): Enable lite-HRNet aggregator. Defaults to False.
            aggregator_min_channels (int, optional): Minimum channels for aggregator. Defaults to 0.
            aggregator_merge_norm (str | None, optional): Aggregator merge normalization. Defaults to None.
            aggregator_use_concat (bool, optional): Use concatenation in aggregator. Defaults to False.
            in_channels (list[int] | int | None, optional): Input channels. Defaults to None.
            in_index (list[int] | int | None, optional): Input index. Defaults to None.
            norm_cfg (dict[str, Any] | None, optional): Normalization configuration. Defaults to None.
            conv_cfg (dict[str, Any] | None, optional): Convolution configuration. Defaults to None.
            input_transform (list | None, optional): Input transform. Defaults to None.
        """
        if enable_aggregator:  # Lite-HRNet aggregator
            if in_channels is None or isinstance(in_channels, int):
                msg = "'in_channels' should be List[int]."
                raise ValueError(msg)
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
            in_channels = max(in_channels[0], aggregator_min_channels)
            input_transform = None
            if isinstance(in_index, list):
                in_index = in_index[0]
        else:
            aggregator = None

        super().__init__(
            *args,
            in_index=in_index,
            norm_cfg=norm_cfg,
            conv_cfg=conv_cfg,
            input_transform=input_transform,
            in_channels=in_channels,
            **kwargs,
        )

        self.aggregator = aggregator
        # re-define variables
        self.in_channels = in_channels  # type: ignore[assignment]
        self.input_transform = input_transform  # type: ignore[assignment]
        self.in_index = in_index  # type: ignore[assignment]

        if self.act_cfg:
            self.convs[-1].act = None  # why we delete last activation?

    def _transform_inputs(self, inputs: list[Tensor]) -> Tensor | list:
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        return self.aggregator(inputs)[0] if self.aggregator is not None else super()._transform_inputs(inputs)
