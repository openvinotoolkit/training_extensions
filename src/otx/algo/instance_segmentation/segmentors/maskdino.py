# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""MaskDINO Instance Segmentation model.

Implementation modified from:
    * https://github.com/IDEA-Research/MaskDINO
    * https://github.com/facebookresearch/Mask2Former
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from otx.algo.instance_segmentation.heads import MaskDINOHead
from otx.algo.instance_segmentation.losses import MaskDINOCriterion
from otx.algo.modules.base_module import BaseModule
from otx.core.data.entity.base import ImageInfo

if TYPE_CHECKING:
    import torch
    from torch import Tensor, nn
    from torchvision import tv_tensors
    from torchvision.models.detection.image_list import ImageList


class MaskDINOModule(BaseModule):
    """Main class for mask classification semantic segmentation architectures.

    Args:
        backbone (nn.Module): backbone network
        sem_seg_head (MaskDINOHead): MaskDINO head including pixel decoder and predictor
        criterion (MaskDINOCriterion): MaskDINO loss criterion
    """

    def __init__(
        self,
        backbone: nn.Module,
        sem_seg_head: MaskDINOHead,
        criterion: MaskDINOCriterion,
    ):
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion

    def forward(
        self,
        images: ImageList,
        imgs_info: list[ImageInfo],
        targets: list[dict[str, Any]] | None = None,
    ) -> dict[str, Tensor] | tuple[list[Tensor], list[torch.LongTensor], list[tv_tensors.Mask]]:
        """Forward pass.

        Args:
            images (ImageList): input images
            imgs_info (list[ImageInfo]): image info (i.e ori_shape) list regarding original images
            targets (list[dict[str, Any]] | None, optional): ground-truth annotations. Defaults to None.

        Returns:
            dict[str, Tensor] | tuple[list[Tensor], list[torch.LongTensor], list[tv_tensors.Mask]]:
                dict[str, Tensor]: loss values
                tuple[list[Tensor], list[torch.LongTensor], list[tv_tensors.Mask]]: prediction results
                    list[Tensor]: bounding boxes and scores with shape [N, 5]
                    list[torch.LongTensor]: labels with shape [N]
                    list[tv_tensors.Mask]: masks with shape [N, H, W]
        """
        features = self.backbone(images.tensors)

        if self.training:
            outputs, mask_dict = self.sem_seg_head(features, targets=targets)
            losses = self.criterion(outputs, targets, mask_dict)
            for k in list(losses.keys()):
                losses[k] *= self.criterion.weight_dict[k]
            return losses

        return self.sem_seg_head.predict(features, imgs_info)

    def export(
        self,
        batch_inputs: torch.Tensor,
        batch_img_metas: list[dict],
    ) -> tuple[list[torch.Tensor], list[torch.LongTensor], list[tv_tensors.Mask]]:
        """Export the model."""
        if len(batch_inputs) != 1:
            msg = "Only support batch size 1 for export"
            raise ValueError(msg)

        features = self.backbone(batch_inputs)
        return self.sem_seg_head.predict(features, batch_img_metas, export=True)
