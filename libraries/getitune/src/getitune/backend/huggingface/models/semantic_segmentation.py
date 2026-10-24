# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Semantic segmentation wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import torch
import torch.nn.functional as f
import transformers
from torchvision import tv_tensors

from getitune.backend.huggingface.models.base import HFModel
from getitune.backend.lightning.models.base import DataInputParams
from getitune.data.entity.sample import PredictionBatch
from getitune.metrics.dice import SegmCallable
from getitune.types.export import TaskLevelExportParameters
from getitune.types.label import LabelInfoTypes, SegLabelInfo
from getitune.types.task import TaskType

if TYPE_CHECKING:
    from torchmetrics import Metric, MetricCollection
    from transformers.utils import ModelOutput

    from getitune.data.entity.sample import SampleBatch

__all__ = ["HFSemanticSegModel"]


class HFSemanticSegModel(HFModel):
    """Semantic segmentation wrapper, e.g. Segformer, BEiT, DPT, UperNet."""

    task: ClassVar[TaskType] = TaskType.SEMANTIC_SEGMENTATION
    hf_auto_class: ClassVar[type] = transformers.AutoModelForSemanticSegmentation
    export_model_type: ClassVar[str] = "Segmentation"
    label_keys: ClassVar[tuple[str, ...]] = ("labels",)
    _onnx_output_names: ClassVar[list[str]] = ["preds"]

    def __init__(
        self,
        checkpoint: str | transformers.PretrainedConfig,
        label_info: LabelInfoTypes,
        *,
        ignore_index: int | None = None,
        soft_threshold: float = 0.5,
        blur_strength: int = -1,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Build the model, defaulting ``semantic_loss_ignore_index`` from the label info.

        Args:
            checkpoint: A Hub repo id or a local checkpoint directory.
            label_info: Label metadata. If a ``SegLabelInfo`` with an
                ``ignore_index`` attribute is passed and *ignore_index* is not
                given explicitly, its value is used.
            ignore_index: Pixel value to exclude from the loss. Defaults to
                255, matching ``DataModule.ignore_index``.
            soft_threshold: Minimum per-pixel class confidence ModelAPI
                requires, embedded in the exported model's metadata.
            blur_strength: ModelAPI soft-mask blur strength, embedded in the
                exported model's metadata; ``-1`` disables blurring.
            **kwargs: Forwarded to :class:`HFModel`.
        """
        dispatched = self._dispatch_label_info(label_info)
        resolved_ignore_index = ignore_index if ignore_index is not None else getattr(dispatched, "ignore_index", 255)
        overrides = dict(kwargs.pop("extra_overrides", None) or {})
        overrides.setdefault("semantic_loss_ignore_index", resolved_ignore_index)
        super().__init__(checkpoint, label_info, extra_overrides=overrides, **kwargs)
        self._ignore_index = resolved_ignore_index
        self._soft_threshold = soft_threshold
        self._blur_strength = blur_strength

    @property
    def _default_preprocessing_params(self) -> dict[str, DataInputParams]:
        """Known-checkpoint preprocessing defaults for semantic segmentation recipes."""
        return {
            "nvidia/segformer-b0-finetuned-ade-512-512": DataInputParams(
                input_size=(512, 512),
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        }

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        return super()._export_parameters.wrap(
            model_type=self.export_model_type,
            task_type="segmentation",
            return_soft_prediction=True,
            soft_threshold=self._soft_threshold,
            blur_strength=self._blur_strength,
        )

    def build_targets(self, batch: SampleBatch) -> dict[str, Any]:
        """Stack Geti's per-sample ``(1, H, W)`` masks into a ``(B, H, W)`` label map (G10)."""
        if batch.masks is None:
            msg = "Semantic segmentation batches need masks."
            raise ValueError(msg)
        masks = torch.vstack([m.as_subclass(torch.Tensor) for m in batch.masks]).long()
        return {"pixel_values": batch.images, "labels": masks}

    def postprocess(self, outputs: ModelOutput, batch: SampleBatch) -> PredictionBatch:
        """Upsample logits to input resolution and take the per-pixel argmax.

        Segformer-family logits come out at a fraction of the input
        resolution (G2) — upsampling before argmax, rather than argmax-then-
        upsample, is what keeps class boundaries aligned with the true pixel
        grid instead of a blocky low-resolution one.
        """
        input_size = batch.images[0].shape[-2:]
        upsampled = f.interpolate(outputs["logits"], size=input_size, mode="bilinear", align_corners=False)
        class_map = upsampled.argmax(dim=1)
        return PredictionBatch(
            images=batch.images,
            imgs_info=batch.imgs_info,
            masks=[tv_tensors.Mask(mask) for mask in class_map.unsqueeze(1)],
        )

    def to_metric_inputs(self, outputs: ModelOutput, batch: SampleBatch) -> dict[str, Any]:
        """Build the ``preds``/``target`` tensors ``SegmCallable`` expects.

        No reprojection is needed here, unlike detection and instance
        segmentation: semantic segmentation validates with ``resize_targets:
        true``, so ground truth is already resized to the model's input
        resolution by the time it reaches this batch.
        """
        if batch.masks is None:
            msg = "Semantic segmentation batches need masks to compute metrics."
            raise ValueError(msg)

        predictions = self.postprocess(outputs, batch)
        if predictions.masks is None:
            msg = "Semantic segmentation postprocess() must always populate masks."
            raise ValueError(msg)

        preds = torch.vstack([mask.as_subclass(torch.Tensor) for mask in predictions.masks]).long()
        target = torch.vstack([mask.as_subclass(torch.Tensor) for mask in batch.masks]).long()
        return {"preds": preds, "target": target}

    def build_default_metric(self) -> Metric | MetricCollection:
        """Dice and mean IoU, the standard semantic-segmentation metrics."""
        seg_label_info = (
            self.label_info
            if isinstance(self.label_info, SegLabelInfo)
            else SegLabelInfo(
                label_names=self.label_info.label_names,
                label_ids=self.label_info.label_ids,
                label_groups=self.label_info.label_groups,
                ignore_index=self._ignore_index,
            )
        )
        return SegmCallable(seg_label_info)

    def forward_for_tracing(self, images: torch.Tensor) -> torch.Tensor:
        """Return per-pixel class probabilities for ONNX/OpenVINO export.

        Same upsample-before-softmax ordering as ``postprocess`` (G2);
        ModelAPI's ``Segmentation`` wrapper expects softmax probabilities,
        not raw logits or an argmax class map.
        """
        input_size = (int(images.shape[-2]), int(images.shape[-1]))
        outputs = self.hf_model(pixel_values=images)
        upsampled = f.interpolate(outputs.logits, size=input_size, mode="bilinear", align_corners=False)
        return upsampled.softmax(dim=1)
