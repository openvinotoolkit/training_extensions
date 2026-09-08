# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""TIMM wrapper model class for getitune."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from getitune.backend.lightning.exporter.base import ModelExporter
from getitune.backend.lightning.exporter.native import LightningModelExporter
from getitune.backend.lightning.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from getitune.backend.lightning.models.classification.backbones.timm import TimmBackbone
from getitune.backend.lightning.models.classification.classifier import ImageClassifier
from getitune.backend.lightning.models.classification.heads import LinearClsHead
from getitune.backend.lightning.models.classification.multiclass_models.base import (
    LightningMulticlassClsModel,
)
from getitune.backend.lightning.models.classification.optimizers import TimmOptimizer
from getitune.backend.lightning.models.classification.utils.pretrained_weights import TimmWeightsLoader
from getitune.backend.lightning.models.classification.utils.timm import get_preprocessing_params
from getitune.backend.lightning.schedulers import LRSchedulerListCallable
from getitune.metrics.accuracy import MultiClassClsMetricCallable
from getitune.types.label import LabelInfoTypes

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from getitune.metrics import MetricCallable
    from getitune.types import PathLike


class TimmModelMulticlassCls(TimmWeightsLoader, LightningMulticlassClsModel):
    """TimmModel for multi-class classification task.

    Args:
        label_info (LabelInfoTypes): Information about the labels.
        data_input_params (DataInputParams | dict | None, optional): The data input parameters
            such as input size and normalization. If None is given,
            default parameters for the specific model will be used.
        model_name (str, optional): Backbone model name for feature extraction. Defaults to "efficientnet_v2_s".
        optimizer (OptimizerCallable, optional): Optimizer for model training. Defaults to DefaultOptimizerCallable.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): Learning rate scheduler.
            Defaults to DefaultSchedulerCallable.
        metric (MetricCallable, optional): Metric for model evaluation. Defaults to MultiClassClsMetricCallable.
        torch_compile (bool, optional): Whether to compile the model using TorchScript. Defaults to False.
        pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
        pretrained_weights (PathLike | None, optional): Path to the pretrained weights file. When None is passed,
            the default pretrained weights will be utilized for fine-tuning. Defaults to None.

    Example:
        >>> model = TimmModelMulticlassCls(
        ...     model_name="tf_efficientnetv2_s.in21k",
        ...     label_info=10,  # Number of classes
        ...     learning_rate=0.0001,
        ... )
    """

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams | dict | None = None,
        model_name: str = "tf_efficientnetv2_s.in21k",
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiClassClsMetricCallable,
        torch_compile: bool = False,
        pretrained: bool = True,
        pretrained_weights: PathLike | None = None,
    ) -> None:
        if isinstance(optimizer, TimmOptimizer):
            optimizer.bind_model_name(model_name)
        super().__init__(
            label_info=label_info,
            data_input_params=data_input_params,
            model_name=model_name,
            freeze_backbone=freeze_backbone,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            pretrained=pretrained,
            pretrained_weights=pretrained_weights,
        )

    def _create_model(self, num_classes: int | None = None) -> nn.Module:
        num_classes = num_classes if num_classes is not None else self.num_classes
        backbone = TimmBackbone(model_name=self.model_name)
        return ImageClassifier(
            backbone=backbone,
            neck=None,
            head=LinearClsHead(
                num_classes=num_classes,
                in_channels=backbone.num_features,
            ),
            loss=nn.CrossEntropyLoss(),
        )

    def forward_for_tracing(self, image: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        if self.explain_mode:
            return self.model(images=image, mode="explain")

        return self.model(images=image, mode="tensor")

    @property
    def _default_preprocessing_params(self) -> DataInputParams | dict[str, DataInputParams]:
        return get_preprocessing_params(backbone_name=self.model_name)

    @property
    def _exporter(self) -> ModelExporter:
        """Force the legacy TorchScript-based ONNX exporter for timm backbones.

        timm's 1700+ architectures frequently use ops (e.g. ``aten.adaptive_max_pool2d``,
        non-contiguous ``transpose().reshape()`` patterns in attention blocks) that the
        newer FX/dynamo-based ``torch.onnx.export`` path cannot yet decompose or trace
        reliably. timm's own official export script (``onnx_export.py``) always uses the
        legacy exporter for exactly this reason; do the same here rather than relying on
        torch's current default.
        """
        exporter = super()._exporter
        modules = ("naflexvit", "nfnet", "volo")
        if not any(s in self.model_name for s in modules):
            assert isinstance(exporter, LightningModelExporter)  # noqa: S101 - internal invariant, not user input
            exporter.onnx_export_configuration["dynamo"] = False
        return exporter
