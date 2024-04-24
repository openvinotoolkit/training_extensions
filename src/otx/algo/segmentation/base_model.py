# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Base segmentation model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch.nn.functional as f
from torch import Tensor, nn

from otx.algo.segmentation.losses import create_criterion

if TYPE_CHECKING:
    from otx.core.data.entity.base import ImageInfo


class BaseSegmNNModel(nn.Module):
    """Base Segmentation Model."""

    def __init__(
        self,
        backbone: nn.Module,
        decode_head: nn.Module,
        criterion_configuration: list[dict[str, str | Any]] | None = None,
    ) -> None:
        """Initializes a SegNext model.

        Args:
            backbone (nn.Module): The backbone of the segmentation model.
            decode_head (nn.Module): The decode head of the segmentation model.
            criterion_configuration (Dict[str, str | Any]): The criterion of the model.
                If None, use CrossEntropyLoss with ignore_index=255.

        Returns:
            None
        """
        super().__init__()

        if criterion_configuration is None:
            criterion_configuration = [{"type": "CrossEntropyLoss", "params": {"ignore_index": 255}}]
        self.backbone = backbone
        self.decode_head = decode_head
        self.criterions = create_criterion(criterion_configuration)

    def forward(
        self,
        inputs: Tensor,
        img_metas: list[ImageInfo] | None = None,
        masks: Tensor | None = None,
        mode: str = "tensor",
    ) -> Tensor:
        """Performs the forward pass of the model.

        Args:
            inputs: Input images to the model.
            img_metas: Image meta information. Defaults to None.
            masks: Ground truth masks for training. Defaults to None.
            mode: The mode of operation. Defaults to "tensor".

        Returns:
            Depending on the mode:
                - If mode is "tensor", returns the model outputs.
                - If mode is "loss", returns a dictionary of output losses.
                - If mode is "predict", returns the predicted outputs.
                - Otherwise, returns the model outputs after interpolation.
        """
        enc_feats = self.backbone(inputs)
        outputs = self.decode_head(enc_feats)

        if mode == "tensor":
            return outputs

        outputs = f.interpolate(outputs, size=inputs.size()[-2:], mode="bilinear", align_corners=True)
        if mode == "loss":
            if masks is None:
                msg = "The masks must be provided for training."
                raise ValueError(msg)
            if img_metas is None:
                msg = "The image meta information must be provided for training."
                raise ValueError(msg)
            # class incremental training
            valid_label_mask = self.get_valid_label_mask(img_metas)
            output_losses = {}
            for criterion in self.criterions:
                valid_label_mask_cfg = {}
                if criterion.name == "loss_ce_ignore":
                    valid_label_mask_cfg["valid_label_mask"] = valid_label_mask
                if criterion.name not in output_losses:
                    output_losses[criterion.name] = criterion(
                        outputs,
                        masks,
                        **valid_label_mask_cfg,
                    )
                else:
                    output_losses[criterion.name] += criterion(
                        outputs,
                        masks,
                        **valid_label_mask_cfg,
                    )
            return output_losses

        if mode == "predict":
            return outputs.argmax(dim=1)

        return outputs

    def get_valid_label_mask(self, img_metas: list[ImageInfo]) -> list[Tensor]:
        """Get valid label mask removing ignored classes to zero mask in a batch.

        Args:
            img_metas (List[dict]): List of image metadata.

        Returns:
            List[torch.Tensor]: List of valid label masks.
        """
        valid_label_mask = []
        for meta in img_metas:
            mask = Tensor([1 for _ in range(self.decode_head.num_classes)])
            if hasattr(meta, "ignored_labels") and meta.ignored_labels:
                mask[meta.ignored_labels] = 0
            valid_label_mask.append(mask)
        return valid_label_mask
