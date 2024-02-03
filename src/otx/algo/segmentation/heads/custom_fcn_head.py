# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Custom FCNHead modules for OTX segmentation model."""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING

import torch
from mmseg.models.decode_heads.fcn_head import FCNHead
from mmseg.models.losses import accuracy
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from torch import Tensor, nn

from otx.algo.utils import IterativeAggregator

if TYPE_CHECKING:
    from mmseg.utils import SampleList


class ClassIncrementalMixin:
    """Mixin for class incremental learning."""

    @typing.no_type_check
    def loss_by_feat(
        self,
        seg_logits: Tensor,
        batch_data_samples: SampleList,
    ) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        img_metas = [data_sample.metainfo for data_sample in batch_data_samples]
        valid_label_mask = self.get_valid_label_mask(img_metas)
        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = {}
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        seg_weight = self.sampler.sample(seg_logits, seg_label) if self.sampler is not None else None
        seg_label = seg_label.squeeze(1)

        losses_decode = [self.loss_decode] if not isinstance(self.loss_decode, nn.ModuleList) else self.loss_decode
        for loss_decode in losses_decode:
            valid_label_mask_cfg = {}
            if loss_decode.loss_name == "loss_ce_ignore":
                valid_label_mask_cfg["valid_label_mask"] = valid_label_mask
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index,
                    **valid_label_mask_cfg,
                )
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index,
                    valid_label_mask=valid_label_mask,
                    **valid_label_mask_cfg,
                )

        loss["acc_seg"] = accuracy(seg_logits, seg_label, ignore_index=self.ignore_index)
        return loss

    @typing.no_type_check
    def get_valid_label_mask(self, img_metas: list[dict]) -> list[torch.Tensor]:
        """Get valid label mask removing ignored classes to zero mask in a batch."""
        valid_label_mask = []
        for meta in img_metas:
            mask = torch.Tensor([1 for _ in range(self.num_classes)])
            if "ignored_labels" in meta and meta["ignored_labels"]:
                mask[meta["ignored_labels"]] = 0
            valid_label_mask.append(mask)
        return valid_label_mask


@MODELS.register_module()
class CustomFCNHead(ClassIncrementalMixin, FCNHead):
    """Custom FCNHead implementation.

    Custom FCNHead supports ignored label for class incremental learning cases.
    """

    def __init__(
        self,
        enable_aggregator: bool = False,
        aggregator_min_channels: int = 0,
        aggregator_merge_norm: str | None = None,
        aggregator_use_concat: bool = False,
        in_channels: list[int] | int | None = None,
        in_index: list[int] | int | None = None,
        norm_cfg: dict | None = None,
        conv_cfg: dict | None = None,
        input_transform: list | None = None,
        *args,
        **kwargs,
    ):
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
        self.in_channels = in_channels
        self.input_transform = input_transform
        self.in_index = in_index

        if self.act_cfg:
            self.convs[-1].with_activation = False
            delattr(self.convs[-1], "activate")

    def _transform_inputs(self, inputs: list[Tensor]) -> Tensor | list:
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        return self.aggregator(inputs)[0] if self.aggregator is not None else super()._transform_inputs(inputs)
