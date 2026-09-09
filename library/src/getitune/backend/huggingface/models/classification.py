# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Classification wrappers: multi-class and multi-label."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import torch
import transformers

from getitune.backend.huggingface.models.base import HFModel
from getitune.backend.lightning.models.base import DataInputParams
from getitune.data.entity.sample import PredictionBatch
from getitune.metrics.accuracy import MultiClassClsMetricCallable, MultiLabelClsMetricCallable
from getitune.types.export import TaskLevelExportParameters
from getitune.types.label import LabelInfoTypes
from getitune.types.task import TaskType

if TYPE_CHECKING:
    from torchmetrics import Metric, MetricCollection
    from transformers.utils import ModelOutput

    from getitune.data.entity.sample import SampleBatch

__all__ = ["HFMulticlassClsModel", "HFMultilabelClsModel"]

_CLASSIFICATION_DEFAULTS: dict[str, DataInputParams] = {
    "google/vit-base-patch16-224": DataInputParams(
        input_size=(224, 224),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
}


class HFMulticlassClsModel(HFModel):
    """Multi-class classification wrapper, e.g. ViT, ConvNeXt, EfficientNet."""

    task: ClassVar[TaskType] = TaskType.MULTI_CLASS_CLS
    hf_auto_class: ClassVar[type] = transformers.AutoModelForImageClassification
    export_model_type: ClassVar[str] = "Classification"
    label_keys: ClassVar[tuple[str, ...]] = ("labels",)
    _onnx_output_names: ClassVar[list[str]] = ["logits"]

    @property
    def _default_preprocessing_params(self) -> dict[str, DataInputParams]:
        """Known-checkpoint preprocessing defaults for multi-class recipes."""
        return _CLASSIFICATION_DEFAULTS

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        return super()._export_parameters.wrap(
            model_type=self.export_model_type,
            task_type="classification",
            multilabel=False,
            output_raw_scores=True,
        )

    def build_targets(self, batch: SampleBatch) -> dict[str, Any]:
        """Stack Geti's per-sample scalar labels into a ``(B,)`` tensor."""
        if batch.labels is None:
            msg = "Classification batches need labels."
            raise ValueError(msg)
        labels = torch.stack([label.long() for label in batch.labels])
        return {"pixel_values": batch.images, "labels": labels}

    def postprocess(self, outputs: ModelOutput, batch: SampleBatch) -> PredictionBatch:
        """Turn logits into a predicted class index and confidence score per image."""
        probabilities = outputs["logits"].softmax(dim=-1)
        scores, labels = probabilities.max(dim=-1)
        return PredictionBatch(
            images=batch.images,
            imgs_info=batch.imgs_info,
            labels=list(labels),
            scores=list(scores),
        )

    def to_metric_inputs(self, outputs: ModelOutput, batch: SampleBatch) -> dict[str, Any]:
        """Build the ``preds``/``target`` tensors ``MultiClassClsMetricCallable`` expects.

        Matches the Lightning convention: predicted class indices compared
        against ground-truth class indices, not raw logits.
        """
        if batch.labels is None:
            msg = "Classification batches need labels to compute metrics."
            raise ValueError(msg)
        predicted_labels = outputs["logits"].argmax(dim=-1)
        target = torch.stack([label.long() for label in batch.labels])
        return {"preds": predicted_labels, "target": target}

    def build_default_metric(self) -> Metric | MetricCollection:
        """Accuracy and macro F1, the standard multi-class classification metrics."""
        return MultiClassClsMetricCallable(self.label_info)

    def forward_for_tracing(self, images: torch.Tensor) -> torch.Tensor:
        """Return raw logits for ONNX/OpenVINO export; ModelAPI applies softmax."""
        return self.hf_model(pixel_values=images).logits


class HFMultilabelClsModel(HFModel):
    """Multi-label classification wrapper.

    Same underlying models as :class:`HFMulticlassClsModel`, with
    ``problem_type="multi_label_classification"`` set on the config so the
    model uses ``BCEWithLogitsLoss`` instead of cross-entropy. Targets are a
    float multi-hot vector rather than a class index.
    """

    task: ClassVar[TaskType] = TaskType.MULTI_LABEL_CLS
    hf_auto_class: ClassVar[type] = transformers.AutoModelForImageClassification
    export_model_type: ClassVar[str] = "Classification"
    label_keys: ClassVar[tuple[str, ...]] = ("labels",)
    _onnx_output_names: ClassVar[list[str]] = ["logits"]

    def __init__(
        self,
        checkpoint: str | transformers.PretrainedConfig,
        label_info: LabelInfoTypes,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Build the model with ``problem_type`` set for multi-label training.

        Args:
            checkpoint: A Hub repo id or a local checkpoint directory.
            label_info: Label metadata.
            **kwargs: Forwarded to :class:`HFModel`.
        """
        overrides = dict(kwargs.pop("extra_overrides", None) or {})
        overrides.setdefault("problem_type", "multi_label_classification")
        super().__init__(checkpoint, label_info, extra_overrides=overrides, **kwargs)

    @property
    def _default_preprocessing_params(self) -> dict[str, DataInputParams]:
        """Known-checkpoint preprocessing defaults for multi-label recipes."""
        return _CLASSIFICATION_DEFAULTS

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        return super()._export_parameters.wrap(
            model_type=self.export_model_type,
            task_type="classification",
            multilabel=True,
            output_raw_scores=True,
        )

    def build_targets(self, batch: SampleBatch) -> dict[str, Any]:
        """Stack Geti's per-sample multi-hot vectors into a ``(B, C)`` float tensor."""
        if batch.labels is None:
            msg = "Classification batches need labels."
            raise ValueError(msg)
        labels = torch.vstack(list(batch.labels)).float()
        return {"pixel_values": batch.images, "labels": labels}

    def postprocess(self, outputs: ModelOutput, batch: SampleBatch) -> PredictionBatch:
        """Turn logits into per-label sigmoid probabilities and a thresholded multi-hot label."""
        scores = outputs["logits"].sigmoid()
        return PredictionBatch(
            images=batch.images,
            imgs_info=batch.imgs_info,
            labels=list((scores >= 0.5).long()),
            scores=list(scores),
        )

    def to_metric_inputs(self, outputs: ModelOutput, batch: SampleBatch) -> dict[str, Any]:
        """Build the ``preds``/``target`` tensors ``MultiLabelClsMetricCallable`` expects.

        Matches the Lightning convention: sigmoid probabilities compared
        against the multi-hot ground truth, not raw logits — the metric's
        own threshold is applied internally.
        """
        if batch.labels is None:
            msg = "Classification batches need labels to compute metrics."
            raise ValueError(msg)
        preds = outputs["logits"].sigmoid()
        target = torch.vstack(list(batch.labels))
        return {"preds": preds, "target": target}

    def build_default_metric(self) -> Metric | MetricCollection:
        """Accuracy and mAP over labels, the standard multi-label classification metrics."""
        return MultiLabelClsMetricCallable(self.label_info)

    def forward_for_tracing(self, images: torch.Tensor) -> torch.Tensor:
        """Return raw logits for ONNX/OpenVINO export; ModelAPI applies sigmoid."""
        return self.hf_model(pixel_values=images).logits
