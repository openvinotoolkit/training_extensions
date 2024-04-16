# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""DINO-V2 model for the OTX classification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from otx.algo.utils.mmconfig import read_mmconfig
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.classification import (
    MulticlassClsBatchDataEntity,
    MulticlassClsBatchPredEntity,
)
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.metrics.accuracy import MultiClassClsMetricCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.classification import OTXMulticlassClsModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.utils.config import inplace_num_classes

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallable


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

    def __init__(
        self,
        num_classes: int,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiClassClsMetricCallable,
        torch_compile: bool = False,
        freeze_backbone: bool = False,
    ) -> None:
        config = read_mmconfig(model_name="dino_v2", subdir_name="multiclass_classification")
        config = inplace_num_classes(cfg=config, num_classes=num_classes)
        config.backbone.frozen = freeze_backbone

        self.config = config

        super().__init__(
            num_classes=num_classes,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

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
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=(1, 3, 224, 224),
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
            resize_mode="standard",
            pad_value=0,
            swap_rgb=False,
            via_onnx=False,
            onnx_export_configuration=None,
            output_names=["logits", "feature_vector", "saliency_map"] if self.explain_mode else None,
        )

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for DinoV2Cls."""
        return {"model_type": "transformer"}

    def forward_for_tracing(self, image: Tensor) -> Tensor | dict[str, Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        return self.model(image)
