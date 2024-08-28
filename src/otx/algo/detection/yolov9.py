# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""YOLOv9 model implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from otx.algo.common.losses import CrossEntropyLoss, L1Loss
from otx.algo.detection.backbones.yolo_v7_v9_backbone import YOLOv9Backbone
from otx.algo.detection.detectors import SingleStageDetector
from otx.algo.detection.heads.yolo_v7_v9_head import YOLOv9Head
from otx.algo.detection.losses import IoULoss, YOLOXCriterion
from otx.algo.detection.necks.yolo_v7_v9_neck import YOLOv9Neck
from otx.core.config.data import TileConfig
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.metrics.fmeasure import MeanAveragePrecisionFMeasureCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.detection import ExplainableOTXDetModel

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallable
    from otx.core.schedulers import LRSchedulerListCallable
    from otx.core.types.label import LabelInfoTypes


PRETRAINED_WEIGHTS: dict[str, str] = {
    # "yolov9-t": "https://github.com/WongKinYiu/YOLO/releases/download/v1.0-alpha/v9-t.pt", # TO BE UPDATED
    "yolov9-s": "https://github.com/WongKinYiu/YOLO/releases/download/v1.0-alpha/v9-s.pt",
    "yolov9-m": "https://github.com/WongKinYiu/YOLO/releases/download/v1.0-alpha/v9-m.pt",
    "yolov9-c": "https://github.com/WongKinYiu/YOLO/releases/download/v1.0-alpha/v9-c.pt",
    # "yolov9-e": "https://github.com/WongKinYiu/YOLO/releases/download/v1.0-alpha/v9-e.pt", # TO BE UPDATED
}


class YOLOv9(ExplainableOTXDetModel):
    """OTX Detection model class for YOLOv9.

    Default input size per model:
        - yolov9-t : (640, 640) # TO BE UPDATED
        - yolov9-s : (640, 640)
        - yolov9-m : (640, 640)
        - yolov9-c : (640, 640)
        - yolov9-e : (640, 640) # TO BE UPDATED
    """

    input_size_multiplier = 32  # TODO (sungchul): need to check
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: tuple[float, float, float] = (255.0, 255.0, 255.0)

    def __init__(
        self,
        model_name: Literal["yolov9-t", "yolov9-s", "yolov9-m", "yolov9-c", "yolov9-e"],
        label_info: LabelInfoTypes,
        input_size: tuple[int, int] = (640, 640),
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MeanAveragePrecisionFMeasureCallable,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
    ) -> None:
        self.load_from: str = PRETRAINED_WEIGHTS[model_name]
        super().__init__(
            model_name=model_name,
            label_info=label_info,
            input_size=input_size,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            tile_config=tile_config,
        )

    def _build_model(self, num_classes: int) -> SingleStageDetector:
        # 'backbone', 'neck', 'head', 'detection', 'auxiliary'
        backbone = YOLOv9Backbone(model_name=self.model_name)
        neck = YOLOv9Neck(model_name=self.model_name)
        bbox_head = YOLOv9Head(model_name=self.model_name, num_classes=num_classes)
        criterion = YOLOXCriterion(
            num_classes=num_classes,
            loss_cls=CrossEntropyLoss(use_sigmoid=True, reduction="sum", loss_weight=1.0),
            loss_bbox=IoULoss(mode="square", eps=1e-16, reduction="sum", loss_weight=5.0),
            loss_obj=CrossEntropyLoss(use_sigmoid=True, reduction="sum", loss_weight=1.0),
            loss_l1=L1Loss(reduction="sum", loss_weight=1.0),
        )
        return SingleStageDetector(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            criterion=criterion,
        )

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        if self.input_size is None:
            msg = f"Input size attribute is not set for {self.__class__}"
            raise ValueError(msg)

        if self.explain_mode:
            msg = "Explainable model export is not supported for YOLOv9 yet."
            raise NotImplementedError(msg)

        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=(1, 3, *self.input_size),
            mean=self.mean,
            std=self.std,
            resize_mode="fit_to_window_letterbox",
            pad_value=114,
            swap_rgb=True,
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
            output_names=None,  # TODO (someone): support XAI
        )
