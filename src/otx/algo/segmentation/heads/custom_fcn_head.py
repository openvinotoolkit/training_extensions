# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Custom FCNHead modules for OTX segmentation model."""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING

import torch
from mmseg.models.decode_heads.fcn_head import FCNHead
from mmseg.models.decode_heads.ham_head import LightHamHead
from mmseg.models.losses import accuracy
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from torch import Tensor, nn

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


@MODELS.register_module()
class CustomLightHamHead(ClassIncrementalMixin, LightHamHead):
    """Custom LightHamHead implementation.

    Custom LightHamHead supports ignored label for class incremental learning cases.
    """
