# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""DINO-V2 model for the OTX classification."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import Tensor, nn

from otx.algo.classification.utils import get_classification_layers
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
from otx.core.types.label import LabelInfoTypes
from otx.utils.utils import get_class_initial_arguments

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from typing_extensions import Self

    from otx.core.metrics import MetricCallable


# TODO(harimkang): Add more types of DINOv2 models. https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md
DINO_BACKBONE_TYPE = Literal["dinov2_vits14_reg"]


class DINOv2(nn.Module):
    """DINO-v2 Model."""

    def __init__(
        self,
        backbone: DINO_BACKBONE_TYPE,
        freeze_backbone: bool,
        head_in_channels: int,
        num_classes: int,
    ):
        super().__init__()
        self._init_args = get_class_initial_arguments()

        ci_data_root = os.environ.get("CI_DATA_ROOT")
        pretrained: bool = True
        if ci_data_root is not None and Path(ci_data_root).exists():
            pretrained = False

        self.backbone = torch.hub.load(
            repo_or_dir="facebookresearch/dinov2",
            model=backbone,
            pretrained=pretrained,
        )

        if ci_data_root is not None and Path(ci_data_root).exists():
            ckpt_filename = f"{backbone}4_pretrain.pth"
            ckpt_path = Path(ci_data_root) / "torch" / "hub" / "checkpoints" / ckpt_filename
            if not ckpt_path.exists():
                msg = f"cannot find weights file: {ckpt_filename}"
                raise FileExistsError(msg)
            self.backbone.load_state_dict(torch.load(ckpt_path))

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

    def forward(self, imgs: torch.Tensor, labels: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """Forward function."""
        feats = self.backbone(imgs)
        logits = self.head(feats)
        if self.training:
            return self.loss(logits, labels)
        return self.softmax(logits)

    def __reduce__(self):
        return (DINOv2, self._init_args)


class DINOv2RegisterClassifier(OTXMulticlassClsModel):
    """DINO-v2 Classification Model with register."""

    def __init__(
        self,
        label_info: LabelInfoTypes,
        backbone: DINO_BACKBONE_TYPE = "dinov2_vits14_reg",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiClassClsMetricCallable,
        torch_compile: bool = False,
        freeze_backbone: bool = False,
    ) -> None:
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone

        super().__init__(
            label_info=label_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    def _create_model(self) -> nn.Module:
        # Get classification_layers for class-incr learning
        sample_model_dict = self._build_model(num_classes=5).state_dict()
        incremental_model_dict = self._build_model(num_classes=6).state_dict()
        self.classification_layers = get_classification_layers(
            sample_model_dict,
            incremental_model_dict,
            prefix="model.",
        )

        return self._build_model(num_classes=self.num_classes)

    def _build_model(self, num_classes: int) -> nn.Module:
        """Create the model."""
        return DINOv2(
            backbone=self.backbone,
            freeze_backbone=self.freeze_backbone,
            # TODO(harimkang): A feature should be added to allow in_channels to adjust based on the arch.
            head_in_channels=384,
            num_classes=num_classes,
        )

    def _customize_inputs(self, entity: MulticlassClsBatchDataEntity) -> dict[str, Any]:
        """Customize the inputs for the model."""
        return {
            "imgs": entity.stacked_images,
            "labels": torch.cat(entity.labels),
            "imgs_info": entity.imgs_info,
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

    def to(self, *args, **kwargs) -> Self:
        """Return a model with specified device."""
        ret = super().to(*args, **kwargs)
        if self.device.type == "xpu":
            msg = f"{type(self).__name__} doesn't support XPU."
            raise RuntimeError(msg)
        return ret
