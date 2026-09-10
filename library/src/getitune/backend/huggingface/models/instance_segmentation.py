# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Instance segmentation wrapper."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, ClassVar

import torch
import torch.nn.functional as f
import transformers
from torchvision import tv_tensors

from getitune.backend.huggingface.models.base import HFModel
from getitune.backend.huggingface.models.utils import (
    _traceable_masks_to_boxes,
    reproject_boxes_to_input_space,
    reproject_masks_to_input_space,
)
from getitune.backend.lightning.models.base import DataInputParams
from getitune.data.entity.sample import PredictionBatch
from getitune.data.utils.structures.mask.mask_util import encode_rle
from getitune.metrics.mean_ap import MaskRLEMeanAPCallable
from getitune.types.export import TaskLevelExportParameters
from getitune.types.task import TaskType

if TYPE_CHECKING:
    from torchmetrics import Metric, MetricCollection
    from transformers.utils import ModelOutput

    from getitune.data.entity.sample import SampleBatch
    from getitune.types.label import LabelInfoTypes

__all__ = ["HFInstSegModel"]


class HFInstSegModel(HFModel):
    """Instance segmentation wrapper for the MaskFormer family.

    Covers Mask2Former, MaskFormer, EoMT, and OneFormer, which all take
    ``mask_labels`` and ``class_labels`` and return
    ``{class_queries_logits, masks_queries_logits}``. Models that instead
    return plain ``labels`` (``DetrForSegmentation``, RF-DETR instance
    segmentation) are out of scope for this wrapper.
    """

    task: ClassVar[TaskType] = TaskType.INSTANCE_SEGMENTATION
    hf_auto_class: ClassVar[type] = transformers.AutoModelForUniversalSegmentation
    export_model_type: ClassVar[str] = "DETRInstSeg"
    label_keys: ClassVar[tuple[str, ...]] = ("mask_labels", "class_labels")
    _onnx_output_names: ClassVar[list[str]] = ["boxes", "labels", "masks"]

    def __init__(
        self,
        checkpoint: str | transformers.PretrainedConfig,
        label_info: LabelInfoTypes,
        *,
        confidence_threshold: float = 0.05,
        iou_threshold: float = 0.5,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Build the model.

        Args:
            checkpoint: A Hub repo id or a local checkpoint directory.
            label_info: Label metadata.
            confidence_threshold: Minimum score ModelAPI keeps an instance
                at, embedded in the exported model's metadata.
            iou_threshold: IoU threshold for ModelAPI's NMS post-processing,
                embedded in the exported model's metadata.
            **kwargs: Forwarded to :class:`HFModel`.
        """
        super().__init__(checkpoint, label_info, **kwargs)
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold

    @property
    def _default_preprocessing_params(self) -> dict[str, DataInputParams]:
        """Known-checkpoint preprocessing defaults for instance segmentation recipes."""
        return {
            "facebook/mask2former-swin-tiny-coco-instance": DataInputParams(
                input_size=(1024, 1024),
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        }

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        # ModelAPI's DETRInstSeg wrapper shifts labels by one and expects a
        # leading placeholder class in the exported label metadata, mirroring
        # RFDETRInst._export_parameters. This is purely an export-time
        # convention; the underlying HF model's own id2label is unaffected.
        #
        # Built via concatenation, not in-place `.insert()`: `LabelInfo.from_num_classes`
        # constructs `label_groups=[label_names]` as the *same* list object, so
        # `.deepcopy()` preserves that aliasing and a naive `.insert()` into both
        # `label_names` and `label_groups[0]` double-inserts the placeholder.
        label_info = copy.deepcopy(self.label_info)
        label_info.label_names = ["getitune_empty_lbl", *label_info.label_names]
        label_info.label_ids = ["None", *label_info.label_ids]
        label_info.label_groups = [["getitune_empty_lbl", *label_info.label_groups[0]], *label_info.label_groups[1:]]

        return super()._export_parameters.wrap(
            model_type=self.export_model_type,
            task_type="instance_segmentation",
            confidence_threshold=self._confidence_threshold,
            iou_threshold=self._iou_threshold,
            label_info=label_info,
        )

    def build_targets(self, batch: SampleBatch) -> dict[str, Any]:
        """Convert Geti's uint8 instance masks and labels to MaskFormer's targets (G9)."""
        if batch.masks is None or batch.labels is None:
            msg = "Instance segmentation batches need masks and labels."
            raise ValueError(msg)
        return {
            "pixel_values": batch.images,
            "mask_labels": [m.as_subclass(torch.Tensor).float() for m in batch.masks],
            "class_labels": [c.long() for c in batch.labels],
        }

    def postprocess(self, outputs: ModelOutput, batch: SampleBatch) -> PredictionBatch:
        """Decode raw outputs into boxes, masks, scores, and labels in the model's input space.

        Kept in the shared input canvas rather than each image's own original
        size, for the same reason as detection: ``to_metric_inputs`` then
        only has to reproject the (simpler) ground truth once, instead of
        rescaling every prediction individually.
        """
        input_size = (int(batch.images[0].shape[-2]), int(batch.images[0].shape[-1]))
        decoded = self._image_processor.post_process_instance_segmentation(  # pyrefly: ignore[missing-attribute]
            outputs,
            threshold=0.0,
            target_sizes=[input_size] * len(batch.images),
            return_binary_maps=True,
        )

        bboxes, masks, labels, scores = [], [], [], []
        for image_result in decoded:
            segmentation = image_result["segmentation"]
            if segmentation.ndim == 2:
                binary_maps = torch.empty((0, *input_size), dtype=torch.bool, device=segmentation.device)
            else:
                binary_maps = segmentation.bool()
            device = binary_maps.device
            bboxes.append(
                tv_tensors.BoundingBoxes(  # pyrefly: ignore[no-matching-overload]
                    _traceable_masks_to_boxes(binary_maps),
                    format=tv_tensors.BoundingBoxFormat.XYXY,
                    canvas_size=input_size,
                )
            )
            masks.append(tv_tensors.Mask(binary_maps))
            segments_info = image_result["segments_info"]
            labels.append(
                torch.tensor([segment["label_id"] for segment in segments_info], dtype=torch.long, device=device)
            )
            scores.append(
                torch.tensor([segment["score"] for segment in segments_info], dtype=torch.float32, device=device)
            )

        return PredictionBatch(
            images=batch.images,
            imgs_info=batch.imgs_info,
            bboxes=bboxes,
            masks=masks,
            labels=labels,
            scores=scores,
        )

    def to_metric_inputs(self, outputs: ModelOutput, batch: SampleBatch) -> dict[str, Any]:
        """Build the ``preds``/``target`` lists ``MaskRLEMeanAPCallable`` expects.

        Unlike detection (which decodes predictions to each image's
        ``ori_shape`` and compares in original image space), instance
        segmentation compares in the model's padded input canvas: the data
        uses ``Resize(keep_aspect_ratio=True)`` (letterbox with padding), and
        a simple ``target_sizes=ori_shape`` rescale does not invert the
        letterbox padding for masks. Decoding predictions to the shared canvas
        and reprojecting ground-truth boxes/masks into that same canvas is the
        exact, efficient path here (G12).
        """
        if batch.bboxes is None or batch.masks is None or batch.labels is None or batch.imgs_info is None:
            msg = "Instance segmentation batches need bboxes, masks, labels, and imgs_info to compute metrics."
            raise ValueError(msg)

        predictions = self.postprocess(outputs, batch)
        if (
            predictions.bboxes is None
            or predictions.masks is None
            or predictions.scores is None
            or predictions.labels is None
        ):
            msg = "Instance segmentation postprocess() must always populate bboxes, masks, scores, and labels."
            raise ValueError(msg)

        input_size = (int(batch.images[0].shape[-2]), int(batch.images[0].shape[-1]))
        preds = [
            {
                "boxes": boxes.as_subclass(torch.Tensor),
                "masks": [encode_rle(mask) for mask in masks.as_subclass(torch.Tensor)],
                "scores": scores,
                "labels": labels,
            }
            for boxes, masks, scores, labels in zip(
                predictions.bboxes, predictions.masks, predictions.scores, predictions.labels, strict=True
            )
        ]
        target = []
        for boxes, masks, labels, img_info in zip(
            batch.bboxes, batch.masks, batch.labels, batch.imgs_info, strict=True
        ):
            if img_info is None:
                msg = "Instance segmentation batches need per-sample img_info to compute metrics."
                raise ValueError(msg)
            target.append(
                {
                    "boxes": reproject_boxes_to_input_space(boxes.as_subclass(torch.Tensor), img_info),
                    "masks": [
                        encode_rle(mask)
                        for mask in reproject_masks_to_input_space(
                            masks.as_subclass(torch.Tensor), img_info, input_size
                        )
                    ],
                    "labels": labels,
                }
            )
        return {"preds": preds, "target": target}

    def build_default_metric(self) -> Metric | MetricCollection:
        """Mean average precision over RLE-encoded masks, the standard instance-seg metric."""
        return MaskRLEMeanAPCallable(self.label_info)

    def forward_for_tracing(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return per-query boxes/labels/masks for ONNX/OpenVINO export.

        The MaskFormer-family models covered by this wrapper (Mask2Former,
        MaskFormer, EoMT, OneFormer) have no box head (G4), so boxes are
        derived from the thresholded mask extents via
        ``_traceable_masks_to_boxes`` — the same trace-safe helper
        ``postprocess`` uses for metrics, but applied to every query at once
        (fixed shape, trace-safe) rather than per decoded segment. The score
        is the top foreground-class probability per query rather than
        ``postprocess``'s ``class_score x mask_quality`` product: computing
        the latter here would need the same per-image segment merging
        ``post_process_instance_segmentation`` does, which is not trace-safe
        (variable segment count). ModelAPI's ``DETRInstSeg`` wrapper expects
        a ``(Q, 5)`` xyxy-plus-score layout per image, hence the label shift
        in ``_export_parameters`` (G21).
        """
        input_size = (int(images.shape[-2]), int(images.shape[-1]))
        outputs = self.hf_model(pixel_values=images)
        masks_logits = f.interpolate(outputs.masks_queries_logits, size=input_size, mode="bilinear")
        masks_probs = masks_logits.sigmoid()
        class_probs = outputs.class_queries_logits.softmax(dim=-1)[..., :-1]
        scores, labels = class_probs.max(dim=-1)

        batch_size, num_queries = masks_probs.shape[:2]
        binary_masks = masks_probs > 0.5
        boxes = _traceable_masks_to_boxes(binary_masks.reshape(batch_size * num_queries, *input_size))
        boxes = boxes.reshape(batch_size, num_queries, 4)
        boxes_with_scores = torch.cat([boxes, scores.unsqueeze(-1)], dim=-1)

        return {"boxes": boxes_with_scores, "labels": labels, "masks": masks_probs}
