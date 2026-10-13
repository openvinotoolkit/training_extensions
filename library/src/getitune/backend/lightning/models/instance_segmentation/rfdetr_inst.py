# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RF-DETR model implementation for Instance Segmentation."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

import torch
from rfdetr.config import (
    RFDETRSeg2XLargeConfig,
    RFDETRSegLargeConfig,
    RFDETRSegMediumConfig,
    RFDETRSegNanoConfig,
    RFDETRSegSmallConfig,
    RFDETRSegXLargeConfig,
)

from getitune.backend.lightning.exporter.base import ModelExporter
from getitune.backend.lightning.exporter.native import LightningModelExporter
from getitune.backend.lightning.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from getitune.backend.lightning.models.common.rfdetr_mixin import RFDETRMixin
from getitune.backend.lightning.models.detection.detectors.rfdetr import RFDETRDetector
from getitune.backend.lightning.models.instance_segmentation.base import LightningInstanceSegModel
from getitune.config.data import TileConfig
from getitune.metrics.fmeasure import MaskRLEMeanAPFMeasureCallable
from getitune.types.export import TaskLevelExportParameters

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from getitune.backend.lightning.schedulers import LRSchedulerListCallable
    from getitune.metrics import MetricCallable
    from getitune.types import PathLike
    from getitune.types.label import LabelInfoTypes


class RFDETRInst(RFDETRMixin, LightningInstanceSegModel):  # pyrefly: ignore[inconsistent-inheritance]
    """getitune Instance Segmentation model class for RF-DETR.

    RF-DETR (Real-time Fast DETR) is a state-of-the-art object detector from Roboflow
    that combines a DINOv2 backbone with a lightweight DETR decoder. This implementation
    adds instance segmentation support with a mask prediction head.

    This implementation uses the rfdetr Python package with RFDETRSeg series for the core model components.

    Args:
        label_info: Information about the labels.
        data_input_params: Parameters for image data preprocessing.
        model_name: Name of the model variant to use.
        optimizer: Callable for the optimizer.
        scheduler: Callable for the learning rate scheduler.
        metric: Callable for the metric.
        multi_scale: Whether to use multi-scale training.
        torch_compile: Whether to use torch compile.
        tile_config: Configuration for tiling.
        max_total_objects_per_batch: Maximum total number of ground-truth objects
            allowed across all images in a training batch. When set, dense
            batches are capped to this budget to prevent OOM errors caused by
            RF-DETR's group-DETR decoder architecture. Objects are kept by
            area (largest first). Set to ``None`` to disable. Only active
            during training.
            Note: Recommended to keep it None to avoid unintended consequences on model performance,
            but it can be set for datasets with many objects per image to avoid OOM errors.
        pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
        pretrained_weights (PathLike | None, optional): Path to the pretrained weights file. When None is passed,
            the default pretrained weights will be utilized for fine-tuning. Defaults to None.

    Note:
        RF-DETR Segmentation uses patch_size=12 with 2 windows for 432x432 input resolution.
    """

    pretrained_urls: ClassVar[dict[str, str]] = {
        "rfdetr_seg_n": "https://storage.googleapis.com/rfdetr/rf-detr-seg-n-ft.pth",
        "rfdetr_seg_s": "https://storage.googleapis.com/rfdetr/rf-detr-seg-s-ft.pth",
        "rfdetr_seg_m": "https://storage.googleapis.com/rfdetr/rf-detr-seg-m-ft.pth",
        "rfdetr_seg_l": "https://storage.googleapis.com/rfdetr/rf-detr-seg-l-ft.pth",
        "rfdetr_seg_xl": "https://storage.googleapis.com/rfdetr/rf-detr-seg-xl-ft.pth",
        "rfdetr_seg_2xl": "https://storage.googleapis.com/rfdetr/rf-detr-seg-2xl-ft.pth",
    }
    _model_config_mapping: ClassVar[dict[str, type]] = {
        "rfdetr_seg_n": RFDETRSegNanoConfig,
        "rfdetr_seg_s": RFDETRSegSmallConfig,
        "rfdetr_seg_m": RFDETRSegMediumConfig,
        "rfdetr_seg_l": RFDETRSegLargeConfig,
        "rfdetr_seg_xl": RFDETRSegXLargeConfig,
        "rfdetr_seg_2xl": RFDETRSeg2XLargeConfig,
    }

    input_size_multiplier = 24

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams | dict | None = None,
        model_name: Literal[
            "rfdetr_seg_n",
            "rfdetr_seg_s",
            "rfdetr_seg_m",
            "rfdetr_seg_l",
            "rfdetr_seg_xl",
            "rfdetr_seg_2xl",
        ] = "rfdetr_seg_m",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MaskRLEMeanAPFMeasureCallable,
        multi_scale: bool = False,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
        max_total_objects_per_batch: int | None = None,
        pretrained: bool = True,
        pretrained_weights: PathLike | None = None,
        export_nms: bool = False,
    ) -> None:
        self.multi_scale = multi_scale
        self.max_total_objects_per_batch = max_total_objects_per_batch
        super().__init__(
            label_info=label_info,
            data_input_params=data_input_params,
            model_name=model_name,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            tile_config=tile_config,
            pretrained=pretrained,
            pretrained_weights=pretrained_weights,
            export_nms=export_nms,
        )

    def _create_model(self, num_classes: int | None = None) -> RFDETRDetector:
        """Create RF-DETR Instance Segmentation model using rfdetr package.

        Args:
            num_classes: Number of classes for detection.

        Returns:
            RFDETRDetector model instance.
        """
        num_classes = num_classes if num_classes is not None else self.num_classes
        return self._build_rfdetr_model(
            num_classes,
            gradient_checkpointing=True,
        )

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Defines parameters required to export a particular model implementation."""
        # RF-DETR-Seg outputs full-image masks (not per-box crops like Mask R-CNN), so we use
        # the "DETRInstSeg" model_type which triggers full-image mask postprocessing in ModelAPI.
        # DETR models use Hungarian matching for one-to-one predictions, but on small datasets
        # near-duplicate boxes can still appear. Use a conservative IoU threshold (0.8) to only
        # suppress almost-identical duplicates without removing valid overlapping detections.
        return super()._export_parameters.wrap(model_type="DETRInstSeg", iou_threshold=0.8)

    @property
    def _exporter(self) -> ModelExporter:
        """Creates ModelExporter object for model export."""
        return LightningModelExporter(
            task_level_export_parameters=self._export_parameters,
            data_input_params=self.data_input_params,
            resize_mode="standard",
            swap_rgb=False,
            via_onnx=False,
            onnx_export_configuration={
                "input_names": ["images"],
                "output_names": ["boxes", "labels", "masks"],
                "dynamic_axes": {"images": {0: "batch"}},
                "dynamo": False,
                "do_constant_folding": True,
                "opset_version": 18,
            },
            output_names=["boxes", "labels", "masks"],
        )

    def forward_for_tracing(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass used for export (returns dict for reliable OV output naming)."""
        boxes_with_scores, labels, masks = self.model.export(  # pyrefly: ignore[not-callable]
            inputs,
            merge_scores=True,
            with_nms=self.export_nms,
        )
        # Scale boxes from normalized [0,1] to pixel coordinates (ModelAPI MaskRCNN expects this)
        h, w = inputs.shape[2], inputs.shape[3]
        scale = torch.tensor([w, h, w, h, 1.0], device=inputs.device)
        boxes_with_scores = boxes_with_scores * scale
        return {"boxes": boxes_with_scores, "labels": labels, "masks": masks}

    @property
    def _default_preprocessing_params(self) -> dict[str, DataInputParams]:  # type: ignore[override]
        """Default preprocessing parameters for RF-DETR segmentation models."""
        imagenet_mean = (0.485, 0.456, 0.406)
        imagenet_std = (0.229, 0.224, 0.225)

        return {
            "rfdetr_seg_n": DataInputParams(
                input_size=(312, 312),
                mean=imagenet_mean,
                std=imagenet_std,
            ),
            "rfdetr_seg_s": DataInputParams(
                input_size=(384, 384),
                mean=imagenet_mean,
                std=imagenet_std,
            ),
            "rfdetr_seg_m": DataInputParams(
                input_size=(432, 432),
                mean=imagenet_mean,
                std=imagenet_std,
            ),
            "rfdetr_seg_l": DataInputParams(
                input_size=(504, 504),
                mean=imagenet_mean,
                std=imagenet_std,
            ),
            "rfdetr_seg_xl": DataInputParams(
                input_size=(624, 624),
                mean=imagenet_mean,
                std=imagenet_std,
            ),
            "rfdetr_seg_2xl": DataInputParams(
                input_size=(768, 768),
                mean=imagenet_mean,
                std=imagenet_std,
            ),
        }
