# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""DINO-V2 model for the OTX classification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.classification import (
    MulticlassClsBatchDataEntity,
    MulticlassClsBatchPredEntity,
)
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.model.entity.classification import OTXMulticlassClsModel
from otx.core.utils.config import inplace_num_classes

if TYPE_CHECKING:
    from omegaconf import DictConfig


class DINOv2(nn.Module):
    """DINO-v2 Model."""

    def __init__(
        self,
        backbone_name: str,
        freeze_backbone: bool,
        head_in_channels: int,
        num_classes: int,
    ):
        super().__init__()
        self.backbone = torch.hub.load(
            repo_or_dir="facebookresearch/dinov2",
            model=backbone_name,
        )

        if freeze_backbone:
            self._freeze_backbone(self.backbone)

        self.head = nn.Linear(
            head_in_channels,
            num_classes,
        )

        self.loss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax()

    def _freeze_backbone(self, backbone: nn.Module) -> None:
        """Freeze the backbone."""
        for _, v in backbone.named_parameters():
            v.requires_grad = False

    def forward(self, imgs: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """Forward function."""
        feats = self.backbone(imgs)
        logits = self.head(feats)
        if self.training:
            return self.loss(logits, labels)
        return self.softmax(logits)


class DINOv2RegisterClassifier(OTXMulticlassClsModel):
    """DINO-v2 Classification Model with register."""

    def __init__(self, num_classes: int, config: DictConfig) -> None:
        config = inplace_num_classes(cfg=config, num_classes=num_classes)
        self.config = config
        super().__init__(num_classes=num_classes)  # create the model

    def _create_model(self) -> nn.Module:
        """Create the model."""
        return DINOv2(
            backbone_name=self.config.backbone.name,
            freeze_backbone=self.config.backbone.frozen,
            head_in_channels=self.config.head.in_channels,
            num_classes=self.config.head.num_classes,
        )

    def _customize_inputs(self, entity: MulticlassClsBatchDataEntity) -> dict[str, Any]:
        """Customize the inputs for the model."""
        return {
            "imgs": entity.stacked_images,
            "labels": torch.cat(entity.labels),
        }

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: MulticlassClsBatchDataEntity,
    ) -> MulticlassClsBatchPredEntity | OTXBatchLossEntity:
        """Customize the outputs for the model."""
        if self.training:
            if not isinstance(outputs, torch.Tensor):
                raise TypeError(outputs)

            losses = OTXBatchLossEntity()
            losses["loss"] = outputs
            return losses

        max_pred_elements, max_pred_idxs = torch.max(outputs, dim=1)
        pred_scores = max_pred_elements
        pred_labels = max_pred_idxs

        scores = torch.unbind(pred_scores, dim=0)
        labels = torch.unbind(pred_labels, dim=0)

        return MulticlassClsBatchPredEntity(
            batch_size=pred_labels.shape[0],
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            labels=labels,
        )

    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Defines parameters required to export a particular model implementation."""
        export_params: dict[str, Any] = {}

        export_params["resize_mode"] = "standard"
        export_params["pad_value"] = 0
        export_params["swap_rgb"] = False
        export_params["via_onnx"] = False
        export_params["input_size"] = (1, 3, 224, 224)
        export_params["onnx_export_configuration"] = None
        export_params["mean"] = [123.675, 116.28, 103.53]
        export_params["std"] = [58.395, 57.12, 57.375]

        parent_parameters = super()._export_parameters
        parent_parameters.update(export_params)

        return parent_parameters

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        return OTXNativeModelExporter(**self._export_parameters)

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for DinoV2Cls."""
        return {"model_type": "transformer"}
