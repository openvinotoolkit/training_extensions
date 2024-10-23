# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Base segmentation model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn.functional as f
from torch import Tensor, nn

from otx.algo.explain.explain_algo import feature_vector_fn

if TYPE_CHECKING:
    from otx.core.data.entity.base import ImageInfo


class BaseSegmModel(nn.Module):
    """Base Segmentation Model.

    Args:
        backbone (nn.Module): The backbone of the segmentation model.
        decode_head (nn.Module): The decode head of the segmentation model.
        criterion (nn.Module, optional): The criterion of the model. Defaults to None.
            If None, use CrossEntropyLoss with ignore_index=255.
    """

    def __init__(
        self,
        backbone: nn.Module,
        decode_head: nn.Module,
        criterion: nn.Module | None = None,
    ) -> None:
        super().__init__()

        self.criterion = nn.CrossEntropyLoss(ignore_index=255) if criterion is None else criterion
        self.backbone = backbone
        self.decode_head = decode_head

    def forward(
        self,
        inputs: Tensor,
        img_metas: list[ImageInfo] | None = None,
        masks: Tensor | None = None,
        mode: str = "tensor",
    ) -> Tensor:
        """Performs the forward pass of the model.

        Args:
            inputs (Tensor): Input images to the model.
            img_metas (list[ImageInfo]): Image meta information. Defaults to None.
            masks (Tensor): Ground truth masks for training. Defaults to None.
            mode (str): The mode of operation. Defaults to "tensor".

        Returns:
            Depending on the mode:
                - If mode is "tensor", returns the model outputs.
                - If mode is "loss", returns a dictionary of output losses.
                - If mode is "predict", returns the predicted outputs.
                - Otherwise, returns the model outputs after interpolation.
        """
        enc_feats, outputs = self.extract_features(inputs)
        outputs = f.interpolate(outputs, size=inputs.size()[2:], mode="bilinear", align_corners=True)

        if mode == "tensor":
            return outputs

        if mode == "loss":
            if masks is None:
                msg = "The masks must be provided for training."
                raise ValueError(msg)
            if img_metas is None:
                msg = "The image meta information must be provided for training."
                raise ValueError(msg)
            return self.calculate_loss(outputs, img_metas, masks, interpolate=False)

        if mode == "predict":
            return outputs.argmax(dim=1)

        if mode == "explain":
            feature_vector = feature_vector_fn(enc_feats)
            return {
                "preds": outputs,
                "feature_vector": feature_vector,
            }

        return outputs

    def extract_features(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Extract features from the backbone and head."""
        enc_feats = self.backbone(inputs)
        return enc_feats, self.decode_head(enc_feats)

    def calculate_loss(
        self,
        model_features: Tensor,
        img_metas: list[ImageInfo],
        masks: Tensor,
        interpolate: bool,
    ) -> Tensor:
        """Calculates the loss of the model.

        Args:
            model_features (Tensor): model outputs of the model.
            img_metas (list[ImageInfo]): Image meta information. Defaults to None.
            masks (Tensor): Ground truth masks for training. Defaults to None.

        Returns:
            Tensor: The loss of the model.
        """
        outputs = (
            f.interpolate(model_features, size=img_metas[0].img_shape, mode="bilinear", align_corners=True)
            if interpolate
            else model_features
        )
        # class incremental training
        valid_label_mask = self.get_valid_label_mask(img_metas)
        output_losses = {}
        valid_label_mask_cfg = {}
        if self.criterion.name == "loss_ce_ignore":
            valid_label_mask_cfg["valid_label_mask"] = valid_label_mask
        if self.criterion.name not in output_losses:
            output_losses[self.criterion.name] = self.criterion(
                outputs,
                masks,
                **valid_label_mask_cfg,
            )
        else:
            output_losses[self.criterion.name] += self.criterion(
                outputs,
                masks,
                **valid_label_mask_cfg,
            )
        return output_losses

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
