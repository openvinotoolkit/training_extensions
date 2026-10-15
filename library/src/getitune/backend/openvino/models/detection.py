# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Class definition for detection model entity used in getitune."""

from __future__ import annotations

import logging as log
from typing import TYPE_CHECKING, Any, Sequence, cast

import numpy as np
import torch
from model_api.tilers import DetectionTiler
from torchvision import tv_tensors

from getitune.backend.openvino.models.base import OVModel
from getitune.backend.openvino.models.utils import rescale_bboxes_to_original
from getitune.data.entity.sample import PredictionBatch, SampleBatch
from getitune.data.entity.tile import TileBatchData
from getitune.metrics import MetricCallable, MetricInput
from getitune.metrics.fmeasure import MeanAveragePrecisionFMeasureCallable
from getitune.types.task import TaskType

if TYPE_CHECKING:
    from model_api.adapters import OpenvinoAdapter
    from model_api.models.utils import DetectionResult
    from torchmetrics import Metric, MetricCollection

    from getitune.data.entity.base import ImageInfo
    from getitune.types import PathLike


class OVDetectionModel(OVModel):
    """OVDetectionModel: Object detection model compatible for OpenVINO IR inference.

    This class is designed to work with OpenVINO IR models or models from the Intel OMZ repository.
    It provides compatibility with the getitune testing pipeline for object detection tasks.

        Initialize the OVDetectionModel.

            model_path (PathLike): Path to the OpenVINO IR model.
            model_type (str): Type of the model (default: "SSD").
            async_inference (bool): Whether to use asynchronous inference (default: True).
            max_num_requests (int | None): Maximum number of inference requests (default: None).
            use_throughput_mode (bool): Whether to use throughput mode (default: True).
            model_api_configuration (dict[str, Any] | None): Configuration for the model API (default: None).
            metric (MetricCallable): Metric callable for evaluation (default: MeanAveragePrecisionFMeasureCallable).
            **kwargs: Additional keyword arguments.
        ...

        Setup the tiler for handling tiled inference tasks.

        This method configures the tiler with the appropriate execution mode
        and disables asynchronous inference as tiling has its own sync/async implementation.
        ...

        Extract hyperparameters from the OpenVINO model adapter.

            model_adapter (OpenvinoAdapter): The adapter to extract model configuration from.

        This method reads the confidence threshold from the model's runtime information (rt_info).
        If unavailable, it logs a warning and sets the confidence threshold to None.
        ...

        Customize the outputs of the model to match the expected format.

            outputs (list[DetectionResult]): List of detection results from the model.
            inputs (SampleBatch): Input batch containing image and metadata.

            PredictionBatch: A batch of predictions including bounding boxes, scores, labels,
            and optionally saliency maps and feature vectors.
        ...

        Prepare inputs for metric computation.

            preds (PredictionBatch): Predicted batch containing bounding boxes, scores, and labels.
            inputs (SampleBatch): Input batch containing ground truth bounding boxes and labels.

            MetricInput: A dictionary with 'preds' and 'target' keys containing
            the predicted and ground truth bounding boxes and labels.
        ...

        Compute evaluation metrics for the model.

            metric (Metric): Metric object used for evaluation.

            dict: A dictionary containing computed metric values.
        ...
    """

    def __init__(
        self,
        model_path: PathLike,
        model_type: str = "SSD",
        async_inference: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = True,
        model_api_configuration: dict[str, Any] | None = None,
        metric: MetricCallable = MeanAveragePrecisionFMeasureCallable,
        **kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path,
            model_type=model_type,
            async_inference=async_inference,
            max_num_requests=max_num_requests,
            use_throughput_mode=use_throughput_mode,
            model_api_configuration=model_api_configuration,
            metric=metric,
        )
        self._task = TaskType.DETECTION

    def _setup_tiler(self) -> None:
        """Setup tiler for tile task."""
        execution_mode = "async" if self.async_inference else "sync"
        # Note: Disable async_inference as tiling has its own sync/async implementation
        self.async_inference = False
        self.model = DetectionTiler(self.model, execution_mode=execution_mode)
        log.info(
            "Enable tiler with tile size: %s                 and overlap: %s",
            self.model.tile_size,
            self.model.tiles_overlap,
        )

    def _get_hparams_from_adapter(self, model_adapter: OpenvinoAdapter) -> None:
        """Reads model configuration from ModelAPI OpenVINO adapter.

        Args:
            model_adapter (OpenvinoAdapter): target adapter to read the config
        """
        if self._is_onnx:
            # For ONNX models, the adapter parses metadata_props into a flat dict.
            metadata = model_adapter.onnx_metadata.get("model_info", {})
            best_confidence_threshold = metadata.get("confidence_threshold", None)
            self.hparams["best_confidence_threshold"] = (
                float(best_confidence_threshold) if best_confidence_threshold is not None else None
            )
        elif model_adapter.model.has_rt_info(["model_info", "confidence_threshold"]):
            best_confidence_threshold = model_adapter.model.get_rt_info(["model_info", "confidence_threshold"]).value
            self.hparams["best_confidence_threshold"] = (
                float(best_confidence_threshold) if best_confidence_threshold is not None else None
            )

    def _customize_outputs(
        self,
        outputs: list[DetectionResult],
        inputs: SampleBatch,
    ) -> PredictionBatch:
        """Customize the outputs of the detection model.

        Args:
            outputs (list[DetectionResult]): A list of detection results containing bounding boxes,
                scores, labels, saliency maps, and feature vectors.
            inputs (SampleBatch): A batch of input data containing images and their metadata.

        Returns:
            PredictionBatch: A batch of predictions containing processed bounding boxes, scores, labels,
            and optionally saliency maps and feature vectors.

        Notes:
            - Adjusts label indices based on whether the first label is "background".
            - Converts bounding boxes to the "XYXY" format and aligns them with the original image shape.
            - Rescales predicted bboxes from model input coordinates (img_shape) to original image
              coordinates (ori_shape) when they differ, ensuring alignment with ground truth targets
              that are not resized (resize_targets=false).
            - Handles optional saliency maps and feature vectors if present in the outputs.
        """
        # add label index
        bboxes = []
        scores = []
        labels = []

        # some OMZ model requires to shift labels
        first_label = (
            self.model.model.get_label_name(0)
            if isinstance(self.model, DetectionTiler)
            else self.model.get_label_name(0)
        )

        label_shift = 1 if first_label == "background" else 0
        if label_shift:
            log.warning("label_shift: %s", label_shift)

        imgs_info = cast("Sequence[ImageInfo]", inputs.imgs_info)
        for i, output in enumerate(outputs):
            img_info = imgs_info[i]
            img_h, img_w = img_info.img_shape
            ori_h, ori_w = img_info.ori_shape

            bboxes_data = torch.as_tensor(output.bboxes, dtype=torch.float32).clone()

            # Rescale predictions from model input coords to original image coords.
            bboxes_data = rescale_bboxes_to_original(
                bboxes_data,
                img_shape=(img_h, img_w),
                ori_shape=(ori_h, ori_w),
                padding=img_info.padding,
                scale_factor=img_info.scale_factor,
            )

            bboxes.append(
                tv_tensors.BoundingBoxes(
                    data=bboxes_data,
                    format="XYXY",
                    canvas_size=(ori_h, ori_w),
                    dtype=torch.float32,
                ),
            )
            # Cast scores to float32 to match the float32 bboxes. ModelAPI returns numpy
            # arrays whose default float dtype is float64; a float64/float32 mismatch makes
            # torchvision NMS fail ("dets should have the same type as scores") during tile merge.
            scores.append(torch.tensor(output.scores.reshape(-1), dtype=torch.float32))
            labels.append(torch.tensor(output.labels.reshape(-1) - label_shift, dtype=torch.long))

        if outputs and outputs[0].saliency_map.size > 1:
            # Squeeze dim 4D => 3D, (1, num_classes, H, W) => (num_classes, H, W)
            predicted_s_maps = [out.saliency_map[0] for out in outputs]

            # Squeeze dim 2D => 1D, (1, internal_dim) => (internal_dim)
            predicted_f_vectors = [out.feature_vector[0] for out in outputs]
            return PredictionBatch(
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=scores,
                bboxes=bboxes,
                labels=labels,
                saliency_map=predicted_s_maps,
                feature_vector=predicted_f_vectors,
            )

        return PredictionBatch(
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=bboxes,
            labels=labels,
        )

    def prepare_metric_inputs(
        self,
        preds: PredictionBatch,  # type: ignore[override]
        inputs: SampleBatch,  # type: ignore[override]
    ) -> MetricInput:
        """Convert prediction and input entities to a format suitable for metric computation.

        Args:
            preds (PredictionBatch): The predicted batch entity containing predicted bboxes.
            inputs (SampleBatch): The input batch entity containing ground truth bboxes.

        Returns:
            MetricInput: A dictionary contains 'preds' and 'target' keys
            corresponding to the predicted and target bboxes for metric evaluation.
        """
        return {
            "preds": [
                {
                    "boxes": bboxes.data,
                    "scores": scores,
                    "labels": labels,
                }
                for bboxes, scores, labels in zip(preds.bboxes, preds.scores, preds.labels)  # type: ignore[arg-type]
            ],
            "target": [
                {
                    "boxes": bboxes.data,
                    "labels": labels,
                }
                for bboxes, labels in zip(inputs.bboxes, inputs.labels)  # type: ignore[arg-type]
            ],
        }

    def compute_metrics(self, metric: Metric | MetricCollection) -> dict:
        """Compute metrics for the model."""
        best_confidence_threshold = self.hparams.get("best_confidence_threshold", None)
        compute_kwargs = {"best_confidence_threshold": best_confidence_threshold}
        return super()._compute_metrics(metric, **compute_kwargs)

    def forward_tiles(self, inputs: TileBatchData) -> PredictionBatch:
        """Run tiled detection inference and merge tile predictions to full-image predictions.

        Delegates tiling to ModelAPI: the datamodule-produced tiles (native-resolution
        crops) and their absolute coordinates are handed to the ModelAPI
        :class:`~model_api.tilers.DetectionTiler`, which runs inference on each tile and
        merges the per-tile detections back to the original image coordinate space
        (offsetting boxes and applying per-class NMS). The merged ModelAPI
        :class:`DetectionResult` is then converted to a getitune :class:`PredictionBatch`.

        Args:
            inputs (TileBatchData): Batch of tiles wrapping the original images.

        Returns:
            PredictionBatch: Merged full-image detection predictions.
        """
        tiler = self._get_tiler()
        label_shift = 1 if tiler.model.get_label_name(0) == "background" else 0
        imgs_info = cast("Sequence[ImageInfo]", inputs.imgs_info)

        images: list[torch.Tensor] = []
        bboxes: list[tv_tensors.BoundingBoxes] = []
        scores: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        for img_tiles, tile_infos, ori_info in zip(
            inputs.batch_tiles,
            inputs.batch_tile_tile_infos,
            imgs_info,
            strict=True,
        ):
            ori_h, ori_w = ori_info.ori_shape
            # ModelAPI expects HWC numpy tiles at their native crop resolution; it resizes
            # each tile to the model input internally and maps predictions back to crop
            # (and then original) coordinates using the supplied tile coordinates.
            np_tiles = [tile.detach().cpu().numpy().transpose(1, 2, 0) for tile in img_tiles]
            coords = [[t.x, t.y, t.x + t.width, t.y + t.height] for t in tile_infos]
            merged: DetectionResult = tiler.predict_tiles(np_tiles, coords, (ori_h, ori_w, 3))

            images.append(torch.empty(3, ori_h, ori_w))
            bboxes.append(
                tv_tensors.BoundingBoxes(
                    torch.as_tensor(np.asarray(merged.bboxes), dtype=torch.float32).reshape(-1, 4),
                    format="XYXY",
                    canvas_size=(ori_h, ori_w),
                ),
            )
            scores.append(torch.as_tensor(np.asarray(merged.scores), dtype=torch.float32).reshape(-1))
            labels.append(torch.as_tensor(np.asarray(merged.labels), dtype=torch.long).reshape(-1) - label_shift)

        return PredictionBatch(
            images=images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=bboxes,
            labels=labels,
        )

    def _get_tiler(self) -> DetectionTiler:
        """Return the ModelAPI detection tiler used to merge tile predictions.

        When the IR was exported with tiling metadata, :meth:`_setup_tiler` has already
        wrapped ``self.model`` in a :class:`~model_api.tilers.DetectionTiler`; that
        instance (configured from the IR metadata) is reused. Otherwise a tiler is
        created on demand and cached, configured from the datamodule ``tile_config``.
        """
        if isinstance(self.model, DetectionTiler):
            return self.model
        if getattr(self, "_fallback_tiler", None) is None:
            self._fallback_tiler = DetectionTiler(
                self.model,
                configuration={
                    "iou_threshold": self.tile_config.iou_threshold,
                    "max_pred_number": self.tile_config.max_num_instances,
                },
                execution_mode="sync",
            )
        return self._fallback_tiler

    def predict_step(self, data_batch: SampleBatch) -> PredictionBatch:
        """Run detection inference and filter by confidence threshold."""
        predictions = self.forward_tiles(data_batch) if isinstance(data_batch, TileBatchData) else self(data_batch)
        threshold = self.hparams.get("best_confidence_threshold", None)
        if not threshold:
            return predictions

        if predictions.scores is None or predictions.bboxes is None or predictions.labels is None:
            return predictions

        filtered_scores: list[torch.Tensor] = []
        filtered_bboxes: list[tv_tensors.BoundingBoxes] = []
        filtered_labels: list[torch.Tensor] = []
        for score, bbox, label in zip(predictions.scores, predictions.bboxes, predictions.labels):
            keep = score > threshold
            filtered_scores.append(score[keep])
            filtered_bboxes.append(
                tv_tensors.BoundingBoxes(data=bbox[keep], format="XYXY", canvas_size=bbox.canvas_size),
            )
            filtered_labels.append(label[keep])

        predictions.scores = filtered_scores
        predictions.bboxes = filtered_bboxes
        predictions.labels = filtered_labels
        return predictions
