# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Class definition for classification model entity used in OTX."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

from otx.core.data.entity.classification import (
    HlabelClsBatchDataEntity,
    HlabelClsBatchPredEntity,
    MulticlassClsBatchDataEntity,
    MulticlassClsBatchPredEntity,
    MultilabelClsBatchDataEntity,
    MultilabelClsBatchPredEntity,
)
from otx.core.metrics import MetricInput
from otx.core.metrics.accuracy import (
    HLabelClsMetricCallble,
    MultiClassClsMetricCallable,
    MultiLabelClsMetricCallable,
)
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable, OTXModel, OVModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.export import TaskLevelExportParameters
from otx.core.types.label import HLabelInfo, LabelInfo, LabelInfoTypes

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from model_api.models.utils import ClassificationResult

    from otx.core.metrics import MetricCallable


class OTXMulticlassClsModel(OTXModel[MulticlassClsBatchDataEntity, MulticlassClsBatchPredEntity]):
    """Base class for the classification models used in OTX."""

    def __init__(
        self,
        label_info: LabelInfoTypes,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiClassClsMetricCallable,
        torch_compile: bool = False,
    ) -> None:
        super().__init__(
            label_info=label_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )
        self.image_size = (1, 3, 224, 224)

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Defines parameters required to export a particular model implementation."""
        return super()._export_parameters.wrap(
            model_type="Classification",
            task_type="classification",
            multilabel=False,
            hierarchical=False,
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: MulticlassClsBatchPredEntity,
        inputs: MulticlassClsBatchDataEntity,
    ) -> MetricInput:
        pred = torch.tensor(preds.labels)
        target = torch.tensor(inputs.labels)
        return {
            "preds": pred,
            "target": target,
        }

    def get_dummy_input(self, batch_size: int = 1) -> MulticlassClsBatchDataEntity:
        """Returns a dummy input for classification model."""
        images = [torch.rand(*self.image_size[1:]) for _ in range(batch_size)]
        labels = [torch.LongTensor([0])] * batch_size
        return MulticlassClsBatchDataEntity(batch_size, images, [], labels=labels)


### NOTE, currently, although we've made the separate Multi-cls, Multi-label classes
### It'll be integrated after H-label classification integration with more advanced design.


class OTXMultilabelClsModel(OTXModel[MultilabelClsBatchDataEntity, MultilabelClsBatchPredEntity]):
    """Multi-label classification models used in OTX."""

    def __init__(
        self,
        label_info: LabelInfoTypes,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiLabelClsMetricCallable,
        torch_compile: bool = False,
    ) -> None:
        super().__init__(
            label_info=label_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )
        self.image_size = (1, 3, 224, 224)

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Defines parameters required to export a particular model implementation."""
        return super()._export_parameters.wrap(
            model_type="Classification",
            task_type="classification",
            multilabel=True,
            hierarchical=False,
            confidence_threshold=0.5,
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: MultilabelClsBatchPredEntity,
        inputs: MultilabelClsBatchDataEntity,
    ) -> MetricInput:
        return {
            "preds": torch.stack(preds.scores),
            "target": torch.stack(inputs.labels),
        }

    def forward_for_tracing(self, image: Tensor) -> Tensor | dict[str, Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        return self.model.forward(image, mode="tensor")

    def get_dummy_input(self, batch_size: int = 1) -> MultilabelClsBatchDataEntity:
        """Returns a dummy input for classification OV model."""
        images = [torch.rand(*self.image_size[1:]) for _ in range(batch_size)]
        labels = [torch.LongTensor([0])] * batch_size
        return MultilabelClsBatchDataEntity(batch_size, images, [], labels=labels)


class OTXHlabelClsModel(OTXModel[HlabelClsBatchDataEntity, HlabelClsBatchPredEntity]):
    """H-label classification models used in OTX."""

    def __init__(
        self,
        label_info: HLabelInfo,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = HLabelClsMetricCallble,
        torch_compile: bool = False,
    ) -> None:
        super().__init__(
            label_info=label_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )
        self.image_size = (1, 3, 224, 224)

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Defines parameters required to export a particular model implementation."""
        return super()._export_parameters.wrap(
            model_type="Classification",
            task_type="classification",
            multilabel=False,
            hierarchical=True,
            confidence_threshold=0.5,
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: HlabelClsBatchPredEntity,
        inputs: HlabelClsBatchDataEntity,
    ) -> MetricInput:
        hlabel_info: HLabelInfo = self.label_info  # type: ignore[assignment]

        if hlabel_info.num_multilabel_classes > 0:
            preds_multiclass = torch.stack(preds.labels)[:, : hlabel_info.num_multiclass_heads]
            preds_multilabel = torch.stack(preds.scores)[:, hlabel_info.num_multiclass_heads :]
            pred_result = torch.cat([preds_multiclass, preds_multilabel], dim=1)
        else:
            pred_result = torch.stack(preds.labels)
        return {
            "preds": pred_result,
            "target": torch.stack(inputs.labels),
        }

    @staticmethod
    def _dispatch_label_info(label_info: LabelInfoTypes) -> LabelInfo:
        if not isinstance(label_info, HLabelInfo):
            raise TypeError(label_info)

        return label_info

    def get_dummy_input(self, batch_size: int = 1) -> HlabelClsBatchDataEntity:
        """Returns a dummy input for classification OV model."""
        images = [torch.rand(*self.image_size[1:]) for _ in range(batch_size)]
        labels = [torch.LongTensor([0])] * batch_size
        return HlabelClsBatchDataEntity(batch_size, images, [], labels=labels)


class OVMulticlassClassificationModel(
    OVModel[MulticlassClsBatchDataEntity, MulticlassClsBatchPredEntity],
):
    """Classification model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the OTX classification model compatible for OTX testing pipeline.
    """

    def __init__(
        self,
        model_name: str,
        model_type: str = "Classification",
        async_inference: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = False,
        model_api_configuration: dict[str, Any] | None = None,
        metric: MetricCallable = MultiClassClsMetricCallable,
        **kwargs,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_type=model_type,
            async_inference=async_inference,
            max_num_requests=max_num_requests,
            use_throughput_mode=use_throughput_mode,
            model_api_configuration=model_api_configuration,
            metric=metric,
        )

    def _customize_outputs(
        self,
        outputs: list[ClassificationResult],
        inputs: MulticlassClsBatchDataEntity,
    ) -> MulticlassClsBatchPredEntity:
        pred_labels = [torch.tensor(out.top_labels[0][0], dtype=torch.long, device=self.device) for out in outputs]
        pred_scores = [torch.tensor(out.top_labels[0][2], device=self.device) for out in outputs]

        if outputs and outputs[0].saliency_map.size != 0:
            # Squeeze dim 4D => 3D, (1, num_classes, H, W) => (num_classes, H, W)
            predicted_s_maps = [out.saliency_map[0] for out in outputs]

            # Squeeze dim 2D => 1D, (1, internal_dim) => (internal_dim)
            predicted_f_vectors = [out.feature_vector[0] for out in outputs]
            return MulticlassClsBatchPredEntity(
                batch_size=len(outputs),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=pred_scores,
                labels=pred_labels,
                saliency_map=predicted_s_maps,
                feature_vector=predicted_f_vectors,
            )

        return MulticlassClsBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=pred_scores,
            labels=pred_labels,
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: MulticlassClsBatchPredEntity,
        inputs: MulticlassClsBatchDataEntity,
    ) -> MetricInput:
        pred = torch.tensor(preds.labels)
        target = torch.tensor(inputs.labels)
        return {
            "preds": pred,
            "target": target,
        }


class OVMultilabelClassificationModel(OVModel[MultilabelClsBatchDataEntity, MultilabelClsBatchPredEntity]):
    """Multilabel classification model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the OTX classification model compatible for OTX testing pipeline.
    """

    def __init__(
        self,
        model_name: str,
        model_type: str = "Classification",
        async_inference: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = True,
        model_api_configuration: dict[str, Any] | None = None,
        metric: MetricCallable = MultiLabelClsMetricCallable,
        **kwargs,
    ) -> None:
        model_api_configuration = model_api_configuration if model_api_configuration else {}
        model_api_configuration.update({"multilabel": True, "confidence_threshold": 0.0})
        super().__init__(
            model_name=model_name,
            model_type=model_type,
            async_inference=async_inference,
            max_num_requests=max_num_requests,
            use_throughput_mode=use_throughput_mode,
            model_api_configuration=model_api_configuration,
            metric=metric,
        )

    def _customize_outputs(
        self,
        outputs: list[ClassificationResult],
        inputs: MultilabelClsBatchDataEntity,
    ) -> MultilabelClsBatchPredEntity:
        pred_scores = [
            torch.tensor([top_label[2] for top_label in out.top_labels], device=self.device) for out in outputs
        ]

        if outputs and outputs[0].saliency_map.size != 0:
            # Squeeze dim 4D => 3D, (1, num_classes, H, W) => (num_classes, H, W)
            predicted_s_maps = [out.saliency_map[0] for out in outputs]

            # Squeeze dim 2D => 1D, (1, internal_dim) => (internal_dim)
            predicted_f_vectors = [out.feature_vector[0] for out in outputs]
            return MultilabelClsBatchPredEntity(
                batch_size=len(outputs),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=pred_scores,
                labels=[],
                saliency_map=predicted_s_maps,
                feature_vector=predicted_f_vectors,
            )

        return MultilabelClsBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=pred_scores,
            labels=[],
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: MultilabelClsBatchPredEntity,
        inputs: MultilabelClsBatchDataEntity,
    ) -> MetricInput:
        return {
            "preds": torch.stack(preds.scores),
            "target": torch.stack(inputs.labels),
        }


class OVHlabelClassificationModel(OVModel[HlabelClsBatchDataEntity, HlabelClsBatchPredEntity]):
    """Hierarchical classification model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the OTX classification model compatible for OTX testing pipeline.
    """

    def __init__(
        self,
        model_name: str,
        model_type: str = "Classification",
        async_inference: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = True,
        model_api_configuration: dict[str, Any] | None = None,
        metric: MetricCallable = HLabelClsMetricCallble,
        **kwargs,
    ) -> None:
        model_api_configuration = model_api_configuration if model_api_configuration else {}
        model_api_configuration.update({"hierarchical": True, "output_raw_scores": True})
        super().__init__(
            model_name=model_name,
            model_type=model_type,
            async_inference=async_inference,
            max_num_requests=max_num_requests,
            use_throughput_mode=use_throughput_mode,
            model_api_configuration=model_api_configuration,
            metric=metric,
        )

    def _customize_outputs(
        self,
        outputs: list[ClassificationResult],
        inputs: HlabelClsBatchDataEntity,
    ) -> HlabelClsBatchPredEntity:
        all_pred_labels = []
        all_pred_scores = []
        for output in outputs:
            logits = output.raw_scores
            predicted_labels = []
            predicted_scores = []
            cls_heads_info = self.model.hierarchical_info["cls_heads_info"]
            for i in range(cls_heads_info["num_multiclass_heads"]):
                logits_begin, logits_end = cls_heads_info["head_idx_to_logits_range"][str(i)]
                head_logits = logits[logits_begin:logits_end]
                j = np.argmax(head_logits)
                predicted_labels.append(j)
                predicted_scores.append(head_logits[j])

            if cls_heads_info["num_multilabel_classes"]:
                logits_begin = cls_heads_info["num_single_label_classes"]
                head_logits = logits[logits_begin:]

                for i in range(head_logits.shape[0]):
                    predicted_scores.append(head_logits[i])
                    if head_logits[i] > self.model.confidence_threshold:
                        predicted_labels.append(1)
                    else:
                        predicted_labels.append(0)

            all_pred_labels.append(torch.tensor(predicted_labels, dtype=torch.long, device=self.device))
            all_pred_scores.append(torch.tensor(predicted_scores, device=self.device))

        if outputs and outputs[0].saliency_map.size != 0:
            # Squeeze dim 4D => 3D, (1, num_classes, H, W) => (num_classes, H, W)
            predicted_s_maps = [out.saliency_map[0] for out in outputs]

            # Squeeze dim 2D => 1D, (1, internal_dim) => (internal_dim)
            predicted_f_vectors = [out.feature_vector[0] for out in outputs]
            return HlabelClsBatchPredEntity(
                batch_size=len(outputs),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=all_pred_scores,
                labels=all_pred_labels,
                saliency_map=predicted_s_maps,
                feature_vector=predicted_f_vectors,
            )

        return HlabelClsBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=all_pred_scores,
            labels=all_pred_labels,
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: HlabelClsBatchPredEntity,
        inputs: HlabelClsBatchDataEntity,
    ) -> MetricInput:
        cls_heads_info = self.model.hierarchical_info["cls_heads_info"]
        num_multilabel_classes = cls_heads_info["num_multilabel_classes"]
        num_multiclass_heads = cls_heads_info["num_multiclass_heads"]
        if num_multilabel_classes > 0:
            preds_multiclass = torch.stack(preds.labels)[:, :num_multiclass_heads]
            preds_multilabel = torch.stack(preds.scores)[:, num_multiclass_heads:]
            pred_result = torch.cat([preds_multiclass, preds_multilabel], dim=1)
        else:
            pred_result = torch.stack(preds.labels)
        return {
            "preds": pred_result,
            "target": torch.stack(inputs.labels),
        }

    def _create_label_info_from_ov_ir(self) -> HLabelInfo:
        ov_model = self.model.get_model()

        if ov_model.has_rt_info(["model_info", "label_info"]):
            serialized = ov_model.get_rt_info(["model_info", "label_info"]).value
            return HLabelInfo.from_json(serialized)

        msg = "Cannot construct LabelInfo from OpenVINO IR. Please check this model is trained by OTX."
        raise ValueError(msg)
