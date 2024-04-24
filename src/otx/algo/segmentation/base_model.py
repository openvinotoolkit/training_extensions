# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Base segmentation model."""

from __future__ import annotations

from typing import Any

import torch.nn.functional as f
from torch import Tensor, nn

from otx.algo.segmentation.losses import create_criterion


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
        images: Tensor,
        masks: Tensor | None = None,
        img_metas: dict[str, Any] | None = None,
        mode: str = "tensor",
    ) -> Tensor:
        """Performs the forward pass of the model.

        Args:
            images: Input images to the model.
            masks: Ground truth masks for training. Defaults to None.
            img_metas: Image meta information. Defaults to None.
            mode: The mode of operation. Defaults to "tensor".

        Returns:
            Depending on the mode:
                - If mode is "tensor", returns the model outputs.
                - If mode is "loss", returns a dictionary of output losses.
                - If mode is "predict", returns the predicted outputs.
                - Otherwise, returns the model outputs after interpolation.
        """
        enc_feats = self.backbone(images)
        outputs = self.decode_head(enc_feats)

        if mode == "tensor":
            return outputs

        outputs = f.interpolate(outputs, size=images.size()[-2:], mode="bilinear", align_corners=True)
        if mode == "loss":
            if masks is None:
                msg = "The masks must be provided for training."
                raise ValueError(msg)
            output_losses = {}
            for criterion in self.criterions:
                output_losses.update({criterion.name: criterion(outputs, masks, img_metas=img_metas)})
            return output_losses

        if mode == "predict":
            return outputs.argmax(dim=1)

        return outputs
