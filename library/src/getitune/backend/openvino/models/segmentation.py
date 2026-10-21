# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Class definition for detection model entity used in getitune."""

from __future__ import annotations

import json
import logging as log
from typing import TYPE_CHECKING, Any

import numpy as np
from model_api.tilers import SemanticSegmentationTiler
from torchvision import tv_tensors

from getitune.backend.openvino.models.base import OVModel
from getitune.data.entity.sample import PredictionBatch, SampleBatch
from getitune.metrics import MetricInput
from getitune.metrics.dice import SegmCallable
from getitune.types.label import SegLabelInfo
from getitune.types.task import TaskType

if TYPE_CHECKING:
    from model_api.models.result import ImageResultWithSoftPrediction

    from getitune.metrics import MetricCallable
    from getitune.types import PathLike


class OVSegmentationModel(OVModel):
    """Semantic segmentation model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the getitune segmentation model compatible for getitune testing pipeline.
    """

    def __init__(
        self,
        model_path: PathLike,
        model_type: str = "Segmentation",
        async_inference: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = True,
        model_api_configuration: dict[str, Any] | None = None,
        metric: MetricCallable = SegmCallable,  # type: ignore[assignment]
        **kwargs,
    ) -> None:
        """Initialize the OVSegmentationModel.

        Args:
            model_path (PathLike): Path to the OpenVINO IR model.
            model_type (str): Type of the model (default: "Segmentation").
            async_inference (bool): Whether to enable asynchronous inference (default: True).
            max_num_requests (int | None): Maximum number of inference requests (default: None).
            use_throughput_mode (bool): Whether to use throughput mode (default: True).
            model_api_configuration (dict[str, Any] | None): Configuration for the model API (default: None).
            metric (MetricCallable): Metric callable for evaluation (default: SegmCallable).
            **kwargs: Additional keyword arguments.
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
        self._task = TaskType.SEMANTIC_SEGMENTATION

    def _setup_tiler(self) -> None:
        """Set up the tiler for tile-based inference.

        This method configures the tiler for semantic segmentation tasks, enabling
        tiled inference with specified tile size and overlap.
        """
        execution_mode = "async" if self.async_inference else "sync"
        # Note: Disable async_inference as tiling has its own sync/async implementation
        self.async_inference = False
        self.model = SemanticSegmentationTiler(self.model, execution_mode=execution_mode)
        log.info(
            "Enable tiler with tile size: %s                 and overlap: %s",
            self.model.tile_size,
            self.model.tiles_overlap,
        )

    def _customize_outputs(
        self,
        outputs: list[ImageResultWithSoftPrediction],
        inputs: SampleBatch,
    ) -> PredictionBatch:
        """Customize the outputs of the model for getitune pipeline.

        Args:
            outputs (list[ImageResultWithSoftPrediction]): List of model outputs with soft predictions.
            inputs (SampleBatch): Input batch containing images and metadata.

        Returns:
            PredictionBatch: Customized prediction batch containing masks and feature vectors.
        """
        masks = [tv_tensors.Mask(np.expand_dims(mask.resultImage, axis=0)) for mask in outputs]
        predicted_f_vectors = (
            [out.feature_vector for out in outputs] if outputs and outputs[0].feature_vector.size != 1 else []
        )
        return PredictionBatch(
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=[],
            masks=masks,
            feature_vector=predicted_f_vectors,
        )

    def prepare_metric_inputs(
        self,
        preds: PredictionBatch,  # type: ignore[override]
        inputs: SampleBatch,  # type: ignore[override]
    ) -> MetricInput:
        """Prepare inputs for metric computation.

        Converts predictions and ground truth inputs into a format suitable for metric evaluation.

        Args:
            preds (PredictionBatch): Predicted segmentation batch containing masks.
            inputs (SampleBatch): Input batch containing ground truth masks.

        Returns:
            MetricInput: A list of dictionaries with 'preds' and 'target' keys for metric evaluation.

        Raises:
            ValueError: If predicted or ground truth masks are not provided.
        """
        if preds.masks is None:
            msg = "The predicted masks are not provided."
            raise ValueError(msg)

        if inputs.masks is None:
            msg = "The input ground truth masks are not provided."
            raise ValueError(msg)

        return [
            {
                "preds": pred_mask,
                "target": target_mask,
            }
            for pred_mask, target_mask in zip(preds.masks, inputs.masks)
        ]

    def _create_label_info_from_model(self) -> SegLabelInfo:
        """Create label information from OpenVINO IR or ONNX model metadata.

        Extracts label information from the model metadata if available.

        Returns:
            SegLabelInfo: Label information extracted from the model.

        Raises:
            ValueError: If label information cannot be constructed from the model metadata.
        """
        if self._is_onnx:
            # For ONNX models, the adapter parses metadata_props into rt_info.
            serialized = self.model.inference_adapter.get_rt_info(["model_info", "label_info"]).astype(str)
            label_info = json.loads(serialized)
            return SegLabelInfo(**label_info)

        # For OV IR models, use the explicit has_rt_info check.
        ov_model = self.model.get_model()
        if ov_model.has_rt_info(["model_info", "label_info"]):
            label_info = json.loads(ov_model.get_rt_info(["model_info", "label_info"]).value)
            return SegLabelInfo(**label_info)

        msg = "Cannot construct LabelInfo from model metadata. Please check this model is trained by getitune."
        raise ValueError(msg)
