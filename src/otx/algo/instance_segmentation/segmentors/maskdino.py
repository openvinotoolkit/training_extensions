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
from otx.algo.modules.conv_module import Conv2dModule
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

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Load model weights from state dict.

        Args:
            state_dict (dict): state dict from pretrained weights.
            prefix (str): A string prefix that is added to the keys in the state_dict when loading nested modules.
            local_metadata (dict): This dictionary contains metadata for the local module, used for versioning.
            strict (bool): If True, it ensures that the keys in state_dict match exactly with the module's parameters.
            missing_keys (list[str]): A list of str containing the missing keys.
            unexpected_keys (list[str]): A list of str containing the unexpected keys.
            error_msgs (list[str]): A list of str containing the error messages.
        """
        backbone_weights_name = []
        backbone_shortcut_weights_name = []
        conv_module_weights_name = []

        conv_modules = ("mask_features", "adapter_1", "layer_1")
        # get original layer names
        for ori_layer_name in state_dict:
            if ori_layer_name.startswith("backbone"):
                if "shortcut" in ori_layer_name:
                    backbone_shortcut_weights_name.append(ori_layer_name)
                else:
                    backbone_weights_name.append(ori_layer_name)
            elif ori_layer_name.startswith("sem_seg_head.pixel_decoder") and any(
                n in ori_layer_name for n in conv_modules
            ):
                conv_module_weights_name.append(ori_layer_name)

        # replace backbone layer names in state_dict
        for new_layer_name in self.backbone.state_dict():
            if "downsample" in new_layer_name:
                ori_layer_name = backbone_shortcut_weights_name.pop(0)
            else:
                ori_layer_name = backbone_weights_name.pop(0)

            # check shape
            if state_dict[ori_layer_name].shape != self.backbone.state_dict()[new_layer_name].shape:
                msg = "Shape mismatch in backbone weights"
                raise ValueError(msg)
            # pop and push
            state_dict["backbone." + new_layer_name] = state_dict.pop(ori_layer_name)

        # replace conv module layer names in state_dict
        for module_name, module in self.sem_seg_head.named_modules():
            if isinstance(module, Conv2dModule):
                for name in module.state_dict():
                    new_layer_name = f"sem_seg_head.{module_name}.{name}"
                    ori_layer_name = conv_module_weights_name.pop(0)
                    if state_dict[ori_layer_name].shape != module.state_dict()[name].shape:
                        msg = "Shape mismatch in conv module weights"
                        raise ValueError(msg)
                    if new_layer_name not in self.state_dict():
                        msg = f"Layer {new_layer_name} not found in the model"
                    # pop and push
                    state_dict[new_layer_name] = state_dict.pop(ori_layer_name)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

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
