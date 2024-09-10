# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""YOLOX model implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from otx.algo.common.losses import CrossEntropyLoss, L1Loss
from otx.algo.detection.backbones import CSPDarknet
from otx.algo.detection.base_models import SingleStageDetector
from otx.algo.detection.heads import YOLOXHead
from otx.algo.detection.losses import IoULoss
from otx.algo.detection.necks import YOLOXPAFPN
from otx.algo.detection.utils.assigners import SimOTAAssigner
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.config.data import TileConfig
from otx.core.data.entity.detection import DetBatchDataEntity
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.metrics.fmeasure import MeanAveragePrecisionFMeasureCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.detection import ExplainableOTXDetModel
from otx.core.types.export import OTXExportFormatType
from otx.core.types.precision import OTXPrecisionType

if TYPE_CHECKING:
    from pathlib import Path

    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallable
    from otx.core.schedulers import LRSchedulerListCallable
    from otx.core.types.label import LabelInfoTypes


class YOLOX(ExplainableOTXDetModel):
    """OTX Detection model class for YOLOX."""

    input_size_multiplier = 32

    def __init__(
        self,
        label_info: LabelInfoTypes,
        input_size: tuple[int, int] = (640, 640),
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MeanAveragePrecisionFMeasureCallable,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
    ) -> None:
        super().__init__(
            label_info=label_info,
            input_size=input_size,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            tile_config=tile_config,
        )

    def _customize_inputs(
        self,
        entity: DetBatchDataEntity,
        pad_size_divisor: int = 32,
        pad_value: int = 114,  # YOLOX uses 114 as pad_value
    ) -> dict[str, Any]:
        return super()._customize_inputs(entity=entity, pad_size_divisor=pad_size_divisor, pad_value=pad_value)

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        if self.input_size is None:
            msg = f"Input size attribute is not set for {self.__class__}"
            raise ValueError(msg)

        swap_rgb = not isinstance(self, YOLOXTINY)  # only YOLOX-TINY uses RGB
        resize_mode: Literal["standard", "fit_to_window_letterbox"] = "fit_to_window_letterbox"
        if self.tile_config.enable_tiler:
            resize_mode = "standard"

        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=(1, 3, *self.input_size),
            mean=self.mean,
            std=self.std,
            resize_mode=resize_mode,
            pad_value=114,
            swap_rgb=swap_rgb,
            via_onnx=True,
            onnx_export_configuration={
                "input_names": ["image"],
                "output_names": ["boxes", "labels"],
                "export_params": True,
                "opset_version": 11,
                "dynamic_axes": {
                    "image": {0: "batch", 2: "height", 3: "width"},
                    "boxes": {0: "batch", 1: "num_dets"},
                    "labels": {0: "batch", 1: "num_dets"},
                },
                "keep_initializers_as_inputs": False,
                "verbose": False,
                "autograd_inlining": False,
            },
            output_names=["bboxes", "labels", "feature_vector", "saliency_map"] if self.explain_mode else None,
        )

    def export(
        self,
        output_dir: Path,
        base_name: str,
        export_format: OTXExportFormatType,
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
        to_exportable_code: bool = False,
    ) -> Path:
        """Export this model to the specified output directory.

        This is required to patch otx.algo.detection.backbones.csp_darknet.Focus.forward to export forward.

        Args:
            output_dir (Path): directory for saving the exported model
            base_name: (str): base name for the exported model file. Extension is defined by the target export format
            export_format (OTXExportFormatType): format of the output model
            precision (OTXExportPrecisionType): precision of the output model

        Returns:
            Path: path to the exported model.
        """
        # patch otx.algo.detection.backbones.csp_darknet.Focus.forward
        orig_focus_forward = self.model.backbone.stem.forward
        try:
            self.model.backbone.stem.forward = self.model.backbone.stem.export
            return super().export(output_dir, base_name, export_format, precision, to_exportable_code)
        finally:
            self.model.backbone.stem.forward = orig_focus_forward

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_det_ckpt(state_dict, add_prefix)


class YOLOXTINY(YOLOX):
    """YOLOX-TINY detector."""

    load_from = (
        "https://storage.openvinotoolkit.org/repositories/"
        "openvino_training_extensions/models/object_detection/v2/yolox_tiny_8x8.pth"
    )
    mean = (123.675, 116.28, 103.53)
    std = (58.395, 57.12, 57.375)

    def __init__(
        self,
        label_info: LabelInfoTypes,
        input_size: tuple[int, int] = (416, 416),
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MeanAveragePrecisionFMeasureCallable,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
    ) -> None:
        super().__init__(
            label_info=label_info,
            input_size=input_size,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            tile_config=tile_config,
        )

    def _build_model(self, num_classes: int) -> SingleStageDetector:
        train_cfg: dict[str, Any] = {"assigner": SimOTAAssigner(center_radius=2.5)}
        test_cfg = {
            "nms": {"type": "nms", "iou_threshold": 0.65},
            "score_thr": 0.01,
            "max_per_img": 100,
        }
        backbone = CSPDarknet(deepen_factor=0.33, widen_factor=0.375)
        neck = YOLOXPAFPN(
            in_channels=[96, 192, 384],
            out_channels=96,
            num_csp_blocks=1,
        )
        bbox_head = YOLOXHead(
            num_classes=num_classes,
            in_channels=96,
            feat_channels=96,
            loss_cls=CrossEntropyLoss(use_sigmoid=True, reduction="sum", loss_weight=1.0),
            loss_bbox=IoULoss(mode="square", eps=1e-16, reduction="sum", loss_weight=5.0),
            loss_obj=CrossEntropyLoss(use_sigmoid=True, reduction="sum", loss_weight=1.0),
            loss_l1=L1Loss(reduction="sum", loss_weight=1.0),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        return SingleStageDetector(backbone, bbox_head, neck=neck, train_cfg=train_cfg, test_cfg=test_cfg)


class YOLOXS(YOLOX):
    """YOLOX-S detector."""

    load_from = (
        "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/"
        "yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth"
    )
    mean = (0.0, 0.0, 0.0)
    std = (1.0, 1.0, 1.0)

    def _build_model(self, num_classes: int) -> SingleStageDetector:
        train_cfg: dict[str, Any] = {"assigner": SimOTAAssigner(center_radius=2.5)}
        test_cfg = {
            "nms": {"type": "nms", "iou_threshold": 0.65},
            "score_thr": 0.01,
            "max_per_img": 100,
        }
        backbone = CSPDarknet(deepen_factor=0.33, widen_factor=0.5)
        neck = YOLOXPAFPN(
            in_channels=[128, 256, 512],
            out_channels=128,
            num_csp_blocks=1,
        )
        bbox_head = YOLOXHead(
            num_classes=num_classes,
            in_channels=128,
            feat_channels=128,
            loss_cls=CrossEntropyLoss(use_sigmoid=True, reduction="sum", loss_weight=1.0),
            loss_bbox=IoULoss(mode="square", eps=1e-16, reduction="sum", loss_weight=5.0),
            loss_obj=CrossEntropyLoss(use_sigmoid=True, reduction="sum", loss_weight=1.0),
            loss_l1=L1Loss(reduction="sum", loss_weight=1.0),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        return SingleStageDetector(backbone, bbox_head, neck=neck, train_cfg=train_cfg, test_cfg=test_cfg)


class YOLOXL(YOLOX):
    """YOLOX-L detector."""

    load_from = (
        "https://download.openmmlab.com/mmdetection/v2.0/yolox/"
        "yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth"
    )
    mean = (0.0, 0.0, 0.0)
    std = (1.0, 1.0, 1.0)

    def _build_model(self, num_classes: int) -> SingleStageDetector:
        train_cfg: dict[str, Any] = {"assigner": SimOTAAssigner(center_radius=2.5)}
        test_cfg = {
            "nms": {"type": "nms", "iou_threshold": 0.65},
            "score_thr": 0.01,
            "max_per_img": 100,
        }
        backbone = CSPDarknet()
        neck = YOLOXPAFPN(in_channels=[256, 512, 1024], out_channels=256)
        bbox_head = YOLOXHead(
            num_classes=num_classes,
            in_channels=256,
            loss_cls=CrossEntropyLoss(use_sigmoid=True, reduction="sum", loss_weight=1.0),
            loss_bbox=IoULoss(mode="square", eps=1e-16, reduction="sum", loss_weight=5.0),
            loss_obj=CrossEntropyLoss(use_sigmoid=True, reduction="sum", loss_weight=1.0),
            loss_l1=L1Loss(reduction="sum", loss_weight=1.0),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        return SingleStageDetector(backbone, bbox_head, neck=neck, train_cfg=train_cfg, test_cfg=test_cfg)


class YOLOXX(YOLOX):
    """YOLOX-X detector."""

    load_from = (
        "https://download.openmmlab.com/mmdetection/v2.0/yolox/"
        "yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth"
    )
    mean = (0.0, 0.0, 0.0)
    std = (1.0, 1.0, 1.0)

    def _build_model(self, num_classes: int) -> SingleStageDetector:
        train_cfg: dict[str, Any] = {"assigner": SimOTAAssigner(center_radius=2.5)}
        test_cfg = {
            "nms": {"type": "nms", "iou_threshold": 0.65},
            "score_thr": 0.01,
            "max_per_img": 100,
        }
        backbone = CSPDarknet(deepen_factor=1.33, widen_factor=1.25)
        neck = YOLOXPAFPN(
            in_channels=[320, 640, 1280],
            out_channels=320,
            num_csp_blocks=4,
        )
        bbox_head = YOLOXHead(
            num_classes=num_classes,
            in_channels=320,
            feat_channels=320,
            loss_cls=CrossEntropyLoss(use_sigmoid=True, reduction="sum", loss_weight=1.0),
            loss_bbox=IoULoss(mode="square", eps=1e-16, reduction="sum", loss_weight=5.0),
            loss_obj=CrossEntropyLoss(use_sigmoid=True, reduction="sum", loss_weight=1.0),
            loss_l1=L1Loss(reduction="sum", loss_weight=1.0),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        return SingleStageDetector(backbone, bbox_head, neck=neck, train_cfg=train_cfg, test_cfg=test_cfg)
