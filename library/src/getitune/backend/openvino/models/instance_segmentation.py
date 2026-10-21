# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Class definition for instance segmentation model entity used in getitune."""

from __future__ import annotations

import logging as log
from typing import TYPE_CHECKING, Any, Sequence, cast

import numpy as np
import torch
from model_api.tilers import InstanceSegmentationTiler
from torchvision import tv_tensors

from getitune.backend.openvino.models.base import OVModel
from getitune.backend.openvino.models.utils import rescale_bboxes_to_original, rescale_masks_to_original
from getitune.data.entity.sample import PredictionBatch, SampleBatch
from getitune.data.entity.tile import TileBatchData
from getitune.data.utils.structures.mask.mask_util import encode_rle
from getitune.metrics import MetricInput
from getitune.metrics.fmeasure import MaskRLEMeanAPFMeasureCallable
from getitune.types.label import LabelInfo
from getitune.types.task import TaskType

if TYPE_CHECKING:
    from model_api.adapters import OpenvinoAdapter
    from model_api.models.utils import InstanceSegmentationResult
    from torchmetrics import Metric, MetricCollection

    from getitune.data.entity.base import ImageInfo
    from getitune.metrics import MetricCallable
    from getitune.types import PathLike


class OVInstanceSegmentationModel(OVModel):
    """Instance segmentation model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the getitune detection model compatible for getitune testing pipeline.
    """

    def __init__(
        self,
        model_path: PathLike,
        model_type: str = "MaskRCNN",
        async_inference: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = True,
        model_api_configuration: dict[str, Any] | None = None,
        metric: MetricCallable = MaskRLEMeanAPFMeasureCallable,
        **kwargs,
    ) -> None:
        """Initialize the instance segmentation model.

        Args:
            model_path (PathLike): Path to the OpenVINO IR model.
            model_type (str): Type of the model (default: "MaskRCNN").
            async_inference (bool): Whether to use asynchronous inference (default: True).
            max_num_requests (int | None): Maximum number of inference requests (default: None).
            use_throughput_mode (bool): Whether to use throughput mode (default: True).
            model_api_configuration (dict[str, Any] | None): Model API configuration (default: None).
            metric (MetricCallable): Metric callable for evaluation (default: MaskRLEMeanAPFMeasureCallable).
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
        self._task = TaskType.INSTANCE_SEGMENTATION

    def _setup_tiler(self) -> None:
        """Set up the tiler for tiled inference.

        This method configures the tiler for the instance segmentation task,
        enabling tiled inference with specified tile size and overlap.
        """
        execution_mode = "async" if self.async_inference else "sync"
        self.async_inference = False  # Disable async_inference as tiling has its own implementation
        self.model = InstanceSegmentationTiler(self.model, execution_mode=execution_mode)
        log.info(
            "Enable tiler with tile size: %s                 and overlap: %s",
            self.model.tile_size,
            self.model.tiles_overlap,
        )

    def _get_hparams_from_adapter(self, model_adapter: OpenvinoAdapter) -> None:
        """Retrieve hyperparameters from the OpenVINO adapter.

        Args:
            model_adapter (OpenvinoAdapter): The OpenVINO adapter to read the configuration from.

        This method reads the confidence threshold from the model's runtime information
        and updates the hyperparameters accordingly.
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
        outputs: list[InstanceSegmentationResult],
        inputs: SampleBatch,
    ) -> PredictionBatch:
        """Customize the model outputs for getitune compatibility.

        Args:
            outputs (list[InstanceSegmentationResult]): Model outputs.
            inputs (SampleBatch): Input data batch.

        Returns:
            PredictionBatch: Customized predictions batch.
        """
        bboxes = []
        scores = []
        labels = []
        masks = []
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

            raw_masks = torch.tensor(output.masks)
            if raw_masks.shape[-2:] == (ori_h, ori_w):
                rescaled_masks = raw_masks
            else:
                rescaled_masks = rescale_masks_to_original(
                    raw_masks,
                    img_shape=(img_h, img_w),
                    ori_shape=(ori_h, ori_w),
                    padding=img_info.padding,
                )
            masks.append(rescaled_masks)

            labels.append(torch.tensor(output.labels.reshape(-1) - 1, dtype=torch.long))

        if outputs and outputs[0].saliency_map:
            predicted_s_maps = []
            for out in outputs:
                image_map = torch.tensor(
                    [s_map for s_map in out.saliency_map if s_map.size > 0],
                    dtype=torch.float32,
                )
                predicted_s_maps.append(image_map)

            predicted_f_vectors = [out.feature_vector[0] for out in outputs]
            return PredictionBatch(
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=scores,
                bboxes=bboxes,
                masks=masks if any(mask.numel() > 0 for mask in masks) else None,
                labels=labels,
                saliency_map=predicted_s_maps,
                feature_vector=predicted_f_vectors,
            )

        return PredictionBatch(
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=bboxes,
            masks=masks if any(mask.numel() > 0 for mask in masks) else None,
            labels=labels,
        )

    def prepare_metric_inputs(
        self,
        preds: PredictionBatch,  # type: ignore[override]
        inputs: SampleBatch,  # type: ignore[override]
    ) -> MetricInput:
        """Prepare inputs for metric computation.

        Converts predictions and ground truth to the format required by the metric
        and caches the ground truth for the current batch.

        Args:
            preds (PredictionBatch): Current batch predictions.
            inputs (SampleBatch): Current batch ground-truth inputs.

        Returns:
            MetricInput: Dictionary containing predictions and ground truth.
        """
        target_info = []

        pred_info = [
            {
                "boxes": preds.bboxes[idx].data if preds.bboxes is not None else torch.empty((0, 4)),
                "masks": [encode_rle(mask) for mask in preds.masks[idx].data]
                if preds.masks is not None and len(preds.masks)
                else torch.empty((0,)),
                "scores": preds.scores[idx],  # type: ignore[index]
                "labels": preds.labels[idx],  # type: ignore[index]
            }
            for idx in range(len(preds.labels))  # type: ignore[arg-type]
        ]

        for idx in range(len(inputs.labels)):  # type: ignore[arg-type]
            if inputs.masks is None:
                msg = "Masks are required for metric computation"
                raise ValueError(msg)
            # An empty (0, H, W) mask tensor is a valid ground truth for images with no
            # instances (e.g. background-only samples); encode it as an empty RLE list
            # instead of treating it as missing masks.
            rles = [encode_rle(mask) for mask in inputs.masks[idx].data] if len(inputs.masks[idx]) else []
            target_info.append(
                {
                    "boxes": inputs.bboxes[idx].data if inputs.bboxes is not None else torch.empty((0, 4)),
                    "masks": rles,
                    "labels": inputs.labels[idx],  # type: ignore[index]
                },
            )
        return {"preds": pred_info, "target": target_info}

    def compute_metrics(self, metric: Metric | MetricCollection) -> dict:
        """Compute evaluation metrics for the model.

        Args:
            metric (Metric | MetricCollection): Metric object to compute the evaluation metrics.

        Returns:
            dict: Computed metrics.
        """
        best_confidence_threshold = self.hparams.get("best_confidence_threshold", None)
        compute_kwargs = {"best_confidence_threshold": best_confidence_threshold}
        return super()._compute_metrics(metric, **compute_kwargs)

    def forward_tiles(self, inputs: TileBatchData) -> PredictionBatch:
        """Run tiled instance segmentation inference and merge tile predictions.

        Delegates tiling to ModelAPI: the datamodule-produced tiles (native-resolution
        crops) and their absolute coordinates are handed to the ModelAPI
        :class:`~model_api.tilers.InstanceSegmentationTiler`, which runs inference on
        each tile and merges the per-tile predictions (boxes, labels, scores and masks)
        back to the original image coordinate space. The merged ModelAPI
        :class:`InstanceSegmentationResult` is then converted to a getitune
        :class:`PredictionBatch`.

        Args:
            inputs (TileBatchData): Batch of tiles wrapping the original images.

        Returns:
            PredictionBatch: Merged full-image instance segmentation predictions.
        """
        tiler = self._get_tiler()
        imgs_info = cast("Sequence[ImageInfo]", inputs.imgs_info)

        images: list[torch.Tensor] = []
        bboxes: list[tv_tensors.BoundingBoxes] = []
        scores: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        masks: list[tv_tensors.Mask] = []
        for img_tiles, tile_infos, ori_info in zip(
            inputs.batch_tiles,
            inputs.batch_tile_tile_infos,
            imgs_info,
            strict=True,
        ):
            ori_h, ori_w = ori_info.ori_shape
            # ModelAPI expects HWC numpy tiles at their native crop resolution; it resizes
            # each tile to the model input internally, produces raw per-instance masks and
            # places them into the full image using the merged, offset boxes.
            np_tiles = [tile.detach().cpu().numpy().transpose(1, 2, 0) for tile in img_tiles]
            coords = [[t.x, t.y, t.x + t.width, t.y + t.height] for t in tile_infos]
            merged: InstanceSegmentationResult = tiler.predict_tiles(np_tiles, coords, (ori_h, ori_w, 3))

            images.append(torch.empty(3, ori_h, ori_w))
            bboxes.append(
                tv_tensors.BoundingBoxes(
                    torch.as_tensor(np.asarray(merged.bboxes), dtype=torch.float32).reshape(-1, 4),
                    format="XYXY",
                    canvas_size=(ori_h, ori_w),
                ),
            )
            scores.append(torch.as_tensor(np.asarray(merged.scores), dtype=torch.float32).reshape(-1))
            # ModelAPI instance labels are 1-based (0 = background); shift to getitune 0-based.
            labels.append(torch.as_tensor(np.asarray(merged.labels), dtype=torch.long).reshape(-1) - 1)
            merged_masks = np.asarray(merged.masks)
            if merged_masks.ndim == 3 and merged_masks.shape[0] > 0:
                masks.append(tv_tensors.Mask(torch.as_tensor(merged_masks, dtype=torch.bool)))
            else:
                masks.append(tv_tensors.Mask(torch.zeros((0, ori_h, ori_w), dtype=torch.bool)))

        return PredictionBatch(
            images=images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=bboxes,
            labels=labels,
            masks=masks if any(mask.numel() > 0 for mask in masks) else None,
        )

    def _get_tiler(self) -> InstanceSegmentationTiler:
        """Return the ModelAPI instance-seg tiler used to merge tile predictions.

        When the IR was exported with tiling metadata, :meth:`_setup_tiler` has already
        wrapped ``self.model`` in a :class:`~model_api.tilers.InstanceSegmentationTiler`;
        that instance (configured from the IR metadata) is reused. Otherwise a tiler is
        created on demand and cached, configured from the datamodule ``tile_config``.
        """
        if isinstance(self.model, InstanceSegmentationTiler):
            return self.model
        if getattr(self, "_fallback_tiler", None) is None:
            self._fallback_tiler = InstanceSegmentationTiler(
                self.model,
                configuration={
                    "iou_threshold": self.tile_config.iou_threshold,
                    "max_pred_number": self.tile_config.max_num_instances,
                },
                execution_mode="sync",
            )
        return self._fallback_tiler

    def predict_step(self, data_batch: SampleBatch) -> PredictionBatch:
        """Run instance segmentation inference and filter by confidence threshold."""
        predictions = self.forward_tiles(data_batch) if isinstance(data_batch, TileBatchData) else self(data_batch)
        threshold = self.hparams.get("best_confidence_threshold", None)
        if not threshold:
            return predictions

        if predictions.scores is None or predictions.bboxes is None or predictions.labels is None:
            return predictions

        filtered_scores: list[torch.Tensor] = []
        filtered_bboxes: list[tv_tensors.BoundingBoxes] = []
        filtered_labels: list[torch.Tensor] = []
        filtered_masks: list = []
        for i, (score, bbox, label) in enumerate(zip(predictions.scores, predictions.bboxes, predictions.labels)):
            keep = score > threshold
            filtered_scores.append(score[keep])
            filtered_bboxes.append(
                tv_tensors.BoundingBoxes(data=bbox[keep], format="XYXY", canvas_size=bbox.canvas_size),
            )
            filtered_labels.append(label[keep])
            if predictions.masks is not None and i < len(predictions.masks):
                filtered_masks.append(predictions.masks[i][keep])

        predictions.scores = filtered_scores
        predictions.bboxes = filtered_bboxes
        predictions.labels = filtered_labels
        if filtered_masks:
            predictions.masks = filtered_masks
        return predictions

    def _create_label_info_from_model(self) -> LabelInfo:
        """Create label information from the OpenVINO IR or ONNX model metadata.

        Reads label information from the model metadata and constructs a LabelInfo object.

        Returns:
            LabelInfo: Label information extracted from the model.
        """
        if self._is_onnx:
            # For ONNX models, the adapter parses metadata_props into rt_info.
            serialized = self.model.inference_adapter.get_rt_info(["model_info", "label_info"]).astype(str)
        else:
            # For OV IR models, use the explicit has_rt_info check.
            ov_model = self.model.get_model()
            if not ov_model.has_rt_info(["model_info", "label_info"]):
                return super()._create_label_info_from_model()
            serialized = ov_model.get_rt_info(["model_info", "label_info"]).value

        ir_label_info = LabelInfo.from_json(serialized)
        if ir_label_info.label_names[0] == "getitune_empty_lbl":
            ir_label_info.label_names.pop(0)
            ir_label_info.label_ids.pop(0)
            ir_label_info.label_groups[0].pop(0)
        return ir_label_info
