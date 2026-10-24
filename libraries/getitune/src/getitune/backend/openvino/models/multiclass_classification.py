# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Class definition for classification model entity used in getitune."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from getitune.backend.openvino.models.base import OVModel
from getitune.data.entity.sample import PredictionBatch, SampleBatch
from getitune.metrics import MetricInput
from getitune.metrics.accuracy import (
    MultiClassClsMetricCallable,
)
from getitune.types.task import TaskType

if TYPE_CHECKING:
    from model_api.models.result import ClassificationResult

    from getitune.metrics import MetricCallable
    from getitune.types import PathLike


class OVMulticlassClassificationModel(OVModel):
    """Classification model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the getitune classification model compatible for getitune testing pipeline.
    """

    def __init__(
        self,
        model_path: PathLike,
        model_type: str = "Classification",
        async_inference: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = False,
        model_api_configuration: dict[str, Any] | None = None,
        metric: MetricCallable = MultiClassClsMetricCallable,
    ) -> None:
        """Initialize the OVMulticlassClassificationModel.

        Args:
            model_path (PathLike): Path to the OpenVINO IR model or model name from Intel OMZ.
            model_type (str): Type of the model. Defaults to "Classification".
            async_inference (bool): Whether to enable asynchronous inference. Defaults to True.
            max_num_requests (int | None): Maximum number of inference requests. Defaults to None.
            use_throughput_mode (bool): Whether to use throughput mode. Defaults to False.
            model_api_configuration (dict[str, Any] | None): Configuration for the model API. Defaults to None.
            metric (MetricCallable): Metric callable for evaluation. Defaults to MultiClassClsMetricCallable.
        """
        super().__init__(
            model_path=model_path,
            model_type=model_type,
            async_inference=async_inference,
            max_num_requests=max_num_requests,
            use_throughput_mode=use_throughput_mode,
            model_api_configuration=model_api_configuration,
            metric=metric,
        )
        self._task = TaskType.MULTI_CLASS_CLS

    def _customize_outputs(
        self,
        outputs: list[ClassificationResult],
        inputs: SampleBatch,
    ) -> PredictionBatch:
        """Customize the outputs of the model for getitune pipeline compatibility.

        Args:
            outputs (list[ClassificationResult]): List of classification results from the model.
            inputs (SampleBatch): Input batch containing images and metadata.

        Returns:
            PredictionBatch: A batch of predictions containing scores, labels,
                and optionally saliency maps and feature vectors.
        """
        pred_labels = [torch.tensor(out.top_labels[0].id, dtype=torch.long) for out in outputs]
        pred_scores = [torch.tensor(out.top_labels[0].confidence) for out in outputs]

        if outputs and outputs[0].saliency_map.size != 0:
            # Squeeze dim 4D => 3D, (1, num_classes, H, W) => (num_classes, H, W)
            predicted_s_maps = [out.saliency_map[0] for out in outputs]

            # Squeeze dim 2D => 1D, (1, internal_dim) => (internal_dim)
            predicted_f_vectors = [out.feature_vector[0] for out in outputs]
            return PredictionBatch(
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=pred_scores,
                labels=pred_labels,
                saliency_map=predicted_s_maps,
                feature_vector=predicted_f_vectors,
            )

        return PredictionBatch(
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=pred_scores,
            labels=pred_labels,
        )

    def prepare_metric_inputs(
        self,
        preds: PredictionBatch,
        inputs: SampleBatch,
    ) -> MetricInput:
        """Prepare inputs for metric computation.

        Converts prediction and input entities into a format suitable for metric evaluation.

        Args:
            preds (PredictionBatch): Predicted batch containing predicted labels and other metadata.
            inputs (SampleBatch): Input batch containing ground truth labels and other metadata.

        Returns:
            MetricInput: A dictionary containing 'preds' and 'target' keys corresponding to predicted and target labels.
        """
        pred = torch.tensor(preds.labels)
        target = torch.tensor(inputs.labels)
        return {
            "preds": pred,
            "target": target,
        }
