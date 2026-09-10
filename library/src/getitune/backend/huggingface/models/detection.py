# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Object detection wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch
import transformers
from torchvision import tv_tensors
from torchvision.ops import box_convert

from getitune.backend.huggingface.models.base import HFModel
from getitune.backend.lightning.models.base import DataInputParams
from getitune.data.entity.sample import PredictionBatch
from getitune.metrics.mean_ap import MeanAPCallable
from getitune.types.export import TaskLevelExportParameters
from getitune.types.task import TaskType

if TYPE_CHECKING:
    from torchmetrics import Metric, MetricCollection
    from transformers.utils import ModelOutput

    from getitune.data.entity.sample import SampleBatch
    from getitune.types.label import LabelInfoTypes

__all__ = ["HFDetectionModel"]


class HFDetectionModel(HFModel):
    """Detection wrapper covering HF's generic object-detection model mapping.

    Targets ``transformers.AutoModelForObjectDetection``, which spans DETR-family
    models (RT-DETRv2, RT-DETR, D-FINE, Deformable/Conditional/DAB-DETR), all of
    which take ``pixel_values`` and return ``{loss, logits, pred_boxes}``.
    """

    task: ClassVar[TaskType] = TaskType.DETECTION
    hf_auto_class: ClassVar[type] = transformers.AutoModelForObjectDetection
    export_model_type: ClassVar[str] = "ssd"
    label_keys: ClassVar[tuple[str, ...]] = ("labels",)
    _onnx_output_names: ClassVar[list[str]] = ["bboxes", "labels", "scores"]

    def __init__(
        self,
        checkpoint: str | transformers.PretrainedConfig,
        label_info: LabelInfoTypes,
        *,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Build the model.

        Args:
            checkpoint: A Hub repo id or a local checkpoint directory.
            label_info: Label metadata.
            confidence_threshold: Minimum score ModelAPI keeps a detection
                at, embedded in the exported model's metadata.
            iou_threshold: IoU threshold for ModelAPI's NMS post-processing,
                embedded in the exported model's metadata.
            **kwargs: Forwarded to :class:`HFModel`.
        """
        super().__init__(checkpoint, label_info, **kwargs)
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        return super()._export_parameters.wrap(
            model_type=self.export_model_type,
            task_type="detection",
            confidence_threshold=self._confidence_threshold,
            iou_threshold=self._iou_threshold,
        )

    @property
    def _default_preprocessing_params(self) -> dict[str, DataInputParams]:
        """Known-checkpoint preprocessing defaults for detection recipes."""
        return {
            "PekingU/rtdetr_v2_r18vd": DataInputParams(
                input_size=(640, 640),
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        }

    def build_targets(self, batch: SampleBatch) -> dict[str, Any]:
        """Convert Geti's absolute-xyxy boxes into normalized cxcywh.

        Geti stores boxes as absolute-pixel xyxy; ``transformers`` detection
        models expect cxcywh normalized by the model's input size. That size
        is just ``batch.images.shape[-2:]`` — Geti has already resized and
        padded the image by the time it reaches here.
        """
        images = batch.images
        if not isinstance(images, torch.Tensor) or batch.bboxes is None or batch.labels is None:
            msg = "Detection batches need stacked images, bboxes, and labels."
            raise ValueError(msg)
        _, _, h, w = images.shape
        labels = []
        for boxes, class_labels in zip(batch.bboxes, batch.labels, strict=True):
            xyxy = boxes.as_subclass(torch.Tensor).float()
            if xyxy.numel():
                x1, y1, x2, y2 = xyxy.unbind(-1)
                cxcywh = torch.stack(
                    [(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h],
                    dim=-1,
                )
            else:
                cxcywh = xyxy.new_zeros((0, 4))
            labels.append({"class_labels": class_labels.long(), "boxes": cxcywh})
        return {"pixel_values": batch.images, "labels": labels}

    def postprocess(self, outputs: ModelOutput, batch: SampleBatch) -> PredictionBatch:
        """Decode raw outputs into boxes, scores, and labels in original image space.

        Predictions are decoded per image to its own ``ori_shape`` (the
        Lightning DETR convention): detection data uses ``resize_targets:
        false``, so ground truth stays in original coordinates and both sides
        of ``to_metric_inputs`` live in the same space with no reprojection.
        Returns every query with no confidence filtering — thresholding for
        user-facing predictions happens in ``HFEngine.predict()``, not here,
        so this same method can also feed ``to_metric_inputs`` for mAP, which
        needs the full unfiltered score distribution.
        """
        if batch.imgs_info is None:
            msg = "Detection batches need imgs_info to decode predictions into original image space."
            raise ValueError(msg)
        target_sizes: list[tuple[int, int]] = []
        for info in batch.imgs_info:
            if info is None:
                msg = "Detection batches need per-sample imgs_info to decode predictions."
                raise ValueError(msg)
            target_sizes.append((int(info.ori_shape[0]), int(info.ori_shape[1])))
        use_focal_loss = getattr(self.hf_model.config, "use_focal_loss", False)
        decoded = self._image_processor.post_process_object_detection(  # pyrefly: ignore[missing-attribute]
            outputs,
            threshold=0.0,
            target_sizes=target_sizes,
            use_focal_loss=use_focal_loss,
        )

        bboxes = [
            tv_tensors.BoundingBoxes(  # pyrefly: ignore[no-matching-overload]
                image_result["boxes"],
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=target_sizes[i],
            )
            for i, image_result in enumerate(decoded)
        ]
        return PredictionBatch(
            images=batch.images,
            imgs_info=batch.imgs_info,
            bboxes=bboxes,
            labels=[cast("torch.Tensor", image_result["labels"]) for image_result in decoded],
            scores=[cast("torch.Tensor", image_result["scores"]) for image_result in decoded],
        )

    def to_metric_inputs(self, outputs: ModelOutput, batch: SampleBatch) -> dict[str, Any]:
        """Build the ``preds``/``target`` lists ``MeanAPCallable`` expects.

        Both predictions and ground truth are in original image coordinates:
        predictions are decoded to ``ori_shape`` by ``postprocess``, and
        detection data is configured with ``resize_targets: false`` so ground
        truth arrives in original coordinates too. No reprojection is needed.
        """
        if batch.bboxes is None or batch.labels is None:
            msg = "Detection batches need bboxes and labels to compute metrics."
            raise ValueError(msg)

        predictions = self.postprocess(outputs, batch)
        if predictions.bboxes is None or predictions.scores is None or predictions.labels is None:
            msg = "Detection postprocess() must always populate bboxes, scores, and labels."
            raise ValueError(msg)

        preds = [
            {"boxes": boxes.as_subclass(torch.Tensor), "scores": scores, "labels": labels}
            for boxes, scores, labels in zip(predictions.bboxes, predictions.scores, predictions.labels, strict=True)
        ]
        target = [
            {"boxes": boxes.as_subclass(torch.Tensor), "labels": labels}
            for boxes, labels in zip(batch.bboxes, batch.labels, strict=True)
        ]
        return {"preds": preds, "target": target}

    def build_default_metric(self) -> Metric | MetricCollection:
        """Mean average precision over boxes, the standard detection metric."""
        return MeanAPCallable(self.label_info)

    def forward_for_tracing(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return per-query boxes/labels/scores for ONNX/OpenVINO export.

        Every query is kept (no top-k or thresholding) so the traced graph
        has a static shape; ModelAPI's SSD-style parser does the filtering
        at inference time. Boxes are normalized XYXY (Lightning's DETR-family
        contract), not the pixel-space boxes ``postprocess`` produces for
        metrics — see design doc 7.1 for why the normalized layout was
        chosen. Background-class handling follows G3.
        """
        outputs = self.hf_model(pixel_values=images)
        if getattr(self.hf_model.config, "use_focal_loss", False):
            probs = outputs.logits.sigmoid()
        else:
            probs = outputs.logits.softmax(dim=-1)[..., :-1]
        scores, labels = probs.max(dim=-1)
        bboxes = box_convert(outputs.pred_boxes, in_fmt="cxcywh", out_fmt="xyxy").clamp(0.0, 1.0)
        return {"bboxes": bboxes, "labels": labels, "scores": scores}
