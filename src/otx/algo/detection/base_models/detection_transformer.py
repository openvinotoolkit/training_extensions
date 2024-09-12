# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Base DETR model implementations."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torchvision.ops import box_convert
from torchvision.tv_tensors import BoundingBoxes

from otx.algo.detection.losses import DetrCriterion
from otx.algo.modules.base_module import BaseModule


class DETR(BaseModule):
    """DETR model.

    Args:
        backbone (nn.Module): Backbone module.
        encoder (nn.Module): Encoder module.
        decoder (nn.Module): Decoder module.
        num_classes (int): Number of classes.
        criterion (nn.Module, optional): Loss function.
            If None then default DetrCriterion is used.
        optimizer_configuration (list[dict], optional): Optimizer configuration.
            Defaults to None.
        multi_scale (list[int], optional): List of image sizes.
            Defaults to None.
        num_top_queries (int, optional): Number of top queries to return.
            Defaults to 300.
        input_size (int, optional): The input size of the model. Default to 640.
    """

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        num_classes: int,
        criterion: nn.Module | None = None,
        optimizer_configuration: list[dict] | None = None,
        multi_scale: list[int] | None = None,
        num_top_queries: int = 300,
        input_size: int = 640,
    ) -> None:
        """DETR model implementation."""
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        if multi_scale is not None:
            self.multi_scale = multi_scale
        else:
            self.multi_scale = [input_size - i * 32 for i in range(-5, 6)] + [input_size] * 2

        self.num_classes = num_classes
        self.num_top_queries = num_top_queries
        self.criterion = (
            criterion
            if criterion is not None
            else DetrCriterion(
                weight_dict={"loss_vfl": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0},
                num_classes=num_classes,
                gamma=2.0,
                alpha=0.75,
            )
        )
        self.optimizer_configuration = optimizer_configuration

    def _forward_features(self, images: Tensor, targets: dict[str, Any] | None = None) -> dict[str, Tensor]:
        images = self.backbone(images)
        images = self.encoder(images)
        return self.decoder(images, targets)

    def forward(self, images: Tensor, targets: dict[str, Any] | None = None) -> dict[str, Tensor] | Tensor:
        """Forward pass of the model."""
        if self.multi_scale and self.training:
            sz = int(np.random.choice(self.multi_scale))
            images = nn.functional.interpolate(images, size=[sz, sz])

        output = self._forward_features(images, targets)
        if self.training:
            return self.criterion(output, targets)
        return output

    def export(
        self,
        batch_inputs: Tensor,
        batch_img_metas: list[dict],
        explain_mode: bool = False,
    ) -> dict[str, Any] | tuple[list[Any], list[Any], list[Any]]:
        """Exports the model."""
        if explain_mode:
            msg = "Explain mode is not supported for DETR models yet."
            raise NotImplementedError(msg)

        return self.postprocess(
            self._forward_features(batch_inputs),
            [meta["img_shape"] for meta in batch_img_metas],
            deploy_mode=True,
        )

    def postprocess(
        self,
        outputs: dict[str, Tensor],
        original_sizes: list[tuple[int, int]],
        deploy_mode: bool = False,
    ) -> dict[str, Tensor] | tuple[list[Tensor], list[Tensor], list[Tensor]]:
        """Post-processes the model outputs.

        Args:
            outputs (dict[str, Tensor]): The model outputs.
            original_sizes (list[tuple[int, int]]): The original image sizes.
            deploy_mode (bool, optional): Whether to run in deploy mode. Defaults to False.

        Returns:
            dict[str, Tensor] | tuple[list[Tensor], list[Tensor], list[Tensor]]: The post-processed outputs.
        """
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]

        # convert bbox to xyxy and rescale back to original size (resize in OTX)
        bbox_pred = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        if not deploy_mode:
            original_size_tensor = torch.tensor(original_sizes).to(bbox_pred.device)
            bbox_pred *= original_size_tensor.flip(1).repeat(1, 2).unsqueeze(1)

        # perform scores computation and gather topk results
        scores = nn.functional.sigmoid(logits)
        scores, index = torch.topk(scores.flatten(1), self.num_top_queries, axis=-1)
        labels = index % self.num_classes
        index = index // self.num_classes
        boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))

        if deploy_mode:
            return {"bboxes": boxes, "labels": labels, "scores": scores}

        scores_list, boxes_list, labels_list = [], [], []

        for sc, bb, ll, original_size in zip(scores, boxes, labels, original_sizes):
            scores_list.append(sc)
            boxes_list.append(
                BoundingBoxes(bb, format="xyxy", canvas_size=original_size),
            )
            labels_list.append(ll.long())

        return scores_list, boxes_list, labels_list
