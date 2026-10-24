# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Class definition for classification model entity used in getitune."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import torch
from torch import Tensor

from getitune.backend.lightning.exporter.base import ModelExporter
from getitune.backend.lightning.exporter.native import LightningModelExporter
from getitune.backend.lightning.models.base import (
    DataInputParams,
    DefaultOptimizerCallable,
    DefaultSchedulerCallable,
    LightningModel,
)
from getitune.backend.lightning.schedulers import LRSchedulerListCallable
from getitune.data.entity.base import BatchLoss
from getitune.data.entity.sample import PredictionBatch, SampleBatch
from getitune.metrics import MetricInput
from getitune.metrics.accuracy import (
    MultiLabelClsMetricCallable,
)
from getitune.types.export import TaskLevelExportParameters
from getitune.types.label import LabelInfoTypes
from getitune.types.task import TaskType

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from getitune.metrics import MetricCallable
    from getitune.types import PathLike


class LightningMultilabelClsModel(LightningModel):
    """Multilabel classification model used in getitune.

    Args:
        label_info (LabelInfoTypes | int | Sequence): Information about the labels used in the model.
            if `Sequence` is given, label info will be constructed from the sequence of label names.
        data_input_params (DataInputParams | dict | None, optional): Parameters for the image data preprocessing.
        model_name (str, optional): Name of the model. Defaults to "multilabel_classification_model".
        optimizer (OptimizerCallable, optional): Callable for the optimizer. Defaults to DefaultOptimizerCallable.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): Callable for the learning rate scheduler.
        Defaults to DefaultSchedulerCallable.
        metric (MetricCallable, optional): Callable for the metric. Defaults to MultiLabelClsMetricCallable.
        torch_compile (bool, optional): Flag to indicate whether to use torch.compile. Defaults to False.
        pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
        pretrained_weights (PathLike | None, optional): Path to the pretrained weights file. When None is passed,
            the default pretrained weights will be utilized for fine-tuning. Defaults to None.
    """

    def __init__(
        self,
        label_info: LabelInfoTypes | Sequence,
        data_input_params: DataInputParams | dict | None = None,
        model_name: str = "multiclass_classification_model",
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiLabelClsMetricCallable,
        torch_compile: bool = False,
        pretrained: bool = True,
        pretrained_weights: PathLike | None = None,
    ) -> None:
        super().__init__(
            label_info=label_info,
            data_input_params=data_input_params,
            model_name=model_name,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            pretrained=pretrained,
            pretrained_weights=pretrained_weights,
        )

        if freeze_backbone:
            classification_layers = self._identify_classification_layers()
            for name, param in self.named_parameters():
                param.requires_grad = name in classification_layers

    def _customize_inputs(self, inputs: SampleBatch) -> dict[str, Any]:
        if self.training:
            mode = "loss"
        elif self.explain_mode:
            mode = "explain"
        else:
            mode = "predict"

        return {
            "images": inputs.images,
            "labels": torch.vstack(inputs.labels),
            "imgs_info": inputs.imgs_info,
            "mode": mode,
        }

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: SampleBatch,
    ) -> PredictionBatch | BatchLoss:
        if self.training:
            return BatchLoss(loss=outputs)

        if self.explain_mode:
            return PredictionBatch(
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                labels=list(outputs["labels"]),
                scores=list(outputs["scores"]),
                saliency_map=[saliency_map.to(torch.float32) for saliency_map in outputs["saliency_map"]],
                feature_vector=[feature_vector.unsqueeze(0) for feature_vector in outputs["feature_vector"]],
            )

        # To list, batch-wise
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs["logits"]
        scores = torch.unbind(logits, 0)

        return PredictionBatch(
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            labels=list(logits.argmax(-1, keepdim=True).unbind(0)),
            scores=list(scores),
        )

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Defines parameters required to export a particular model implementation."""
        return super()._export_parameters.wrap(
            model_type="Classification",
            task_type="classification",
            multilabel=True,
            confidence_threshold=0.5,
            output_raw_scores=True,
        )

    @property
    def _exporter(self) -> ModelExporter:
        """Creates ModelExporter object that can export the model."""
        return LightningModelExporter(
            task_level_export_parameters=self._export_parameters,
            data_input_params=self.data_input_params,
            resize_mode="standard",
            pad_value=0,
            swap_rgb=False,
            via_onnx=False,
            onnx_export_configuration=None,
            output_names=["logits", "feature_vector", "saliency_map"] if self.explain_mode else None,
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: PredictionBatch,
        inputs: SampleBatch,
    ) -> MetricInput:
        return {
            "preds": preds.scores,
            "target": inputs.labels,
        }

    def forward_for_tracing(self, image: Tensor) -> Tensor | dict[str, Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        return self.model.forward(image)

    def get_dummy_input(self, batch_size: int = 1) -> SampleBatch:  # type: ignore[override]
        """Returns a dummy input for classification model."""
        if self.data_input_params.input_size is None:
            msg = "input_size should not be None."
            raise ValueError(msg)
        images = torch.stack([torch.rand(3, *self.data_input_params.input_size) for _ in range(batch_size)])
        labels = [torch.LongTensor([0])] * batch_size
        return SampleBatch(images=images, labels=labels)

    def forward_explain(self, inputs: SampleBatch) -> PredictionBatch:
        """Model forward explain function."""
        outputs = self.model(images=inputs.images, mode="explain")

        return PredictionBatch(
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            labels=list(outputs["preds"]),
            scores=list(outputs["scores"]),
            saliency_map=[saliency_map.to(torch.float32) for saliency_map in outputs["saliency_map"]],
            feature_vector=[feature_vector.unsqueeze(0) for feature_vector in outputs["feature_vector"]],
        )

    @property
    def task(self) -> TaskType:
        """Return task type."""
        return TaskType.MULTI_LABEL_CLS

    @property
    def _default_preprocessing_params(self) -> DataInputParams | dict[str, DataInputParams]:
        return DataInputParams(input_size=(224, 224), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
