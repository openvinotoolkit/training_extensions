# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for keypoint detection model entity used in getitune."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from getitune.backend.openvino.models.base import OVModel
from getitune.data.entity.sample import PredictionBatch, SampleBatch
from getitune.metrics import MetricCallable, MetricInput
from getitune.metrics.pck import PCKMeasureCallable
from getitune.types.task import TaskType

if TYPE_CHECKING:
    from model_api.models.result import DetectedKeypoints
    from torchmetrics import Metric, MetricCollection

    from getitune.types import PathLike


class OVKeypointDetectionModel(OVModel):
    """Keypoint detection model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the getitune keypoint detection model compatible for getitune testing pipeline.
    """

    def __init__(
        self,
        model_path: PathLike,
        model_type: str = "keypoint_detection",
        async_inference: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = True,
        model_api_configuration: dict[str, Any] | None = None,
        metric: MetricCallable = PCKMeasureCallable,
    ) -> None:
        """Initialize the keypoint detection model.

        Args:
            model_path (PathLike): Path to the OpenVINO IR model.
            model_type (str): Type of the model. Defaults to "keypoint_detection".
            async_inference (bool): Whether to enable asynchronous inference. Defaults to True.
            max_num_requests (int | None): Maximum number of inference requests. Defaults to None.
            use_throughput_mode (bool): Whether to enable throughput mode. Defaults to True.
            model_api_configuration (dict[str, Any] | None): Configuration for the model API. Defaults to None.
            metric (MetricCallable): Metric callable for evaluation. Defaults to PCKMeasureCallable.
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
        self._task = TaskType.KEYPOINT_DETECTION

    def _customize_outputs(
        self,
        outputs: list[DetectedKeypoints],
        inputs: SampleBatch,
    ) -> PredictionBatch:
        """Customize the outputs of the model for keypoint detection.

        Args:
            outputs (list[DetectedKeypoints]): List of detected keypoints from the model.
            inputs (SampleBatch): Input batch containing images and metadata.

        Returns:
            PredictionBatch: A batch containing processed keypoints, scores, and other metadata.
        """
        keypoints = []
        scores = []
        # default visibility threshold
        visibility_threshold = 0.5
        for output in outputs:
            kps = torch.as_tensor(output.keypoints)
            score = torch.as_tensor(output.scores)
            visible_keypoints = torch.cat([kps, score.unsqueeze(1) > visibility_threshold], dim=1)
            keypoints.append(visible_keypoints)
            scores.append(score)

        return PredictionBatch(
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            keypoints=keypoints,
            scores=scores,
            bboxes=[],
            labels=[],
        )

    def compute_metrics(self, metric: Metric | MetricCollection) -> dict:
        """Compute evaluation metrics for the keypoint detection model.

        Args:
            metric (Metric | MetricCollection): Metric object used for evaluation.

        Returns:
            dict: A dictionary containing computed metric values.
        """
        metric.input_size = (self.model.h, self.model.w)  # type: ignore[union-attr]
        return super()._compute_metrics(metric)

    def prepare_metric_inputs(  # type: ignore[override]
        self,
        preds: PredictionBatch,
        inputs: SampleBatch,
    ) -> MetricInput:
        """Prepare inputs for metric computation.

        Converts prediction and input entities to a format suitable for metric evaluation.

        Args:
            preds (PredictionBatch): The predicted batch entity containing predicted keypoints.
            inputs (SampleBatch): The input batch entity containing ground truth keypoints.

        Returns:
            MetricInput: A dictionary containing 'preds' and 'target' keys
            corresponding to the predicted and target keypoints for metric evaluation.

        Raises:
            ValueError: If ground truth keypoints, predicted keypoints, or scores are missing,
                        or if the number of predicted and ground truth keypoints does not match.
        """
        if inputs.keypoints is None:
            msg = "The input ground truth keypoints are not provided."
            raise ValueError(msg)

        if preds.keypoints is None or preds.scores is None:
            msg = "The predicted keypoints or scores are not provided."
            raise ValueError(msg)

        if len(preds.keypoints) != len(inputs.keypoints):
            msg = "The number of predicted keypoints and ground truth keypoints does not match."
            raise ValueError(msg)

        return {
            "preds": [
                {
                    "keypoints": kpt[:, :2],
                    "scores": score,
                }
                for kpt, score in zip(preds.keypoints, preds.scores)
            ],
            "target": [
                {
                    "keypoints": kpt[:, :2],
                    "keypoints_visible": kpt[:, 2],
                }
                for kpt in inputs.keypoints
            ],
        }
