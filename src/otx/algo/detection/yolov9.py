# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""YOLOv9 model implementations."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

from otx.algo.detection.backbones import GELAN
from otx.algo.detection.detectors import SingleStageDetector
from otx.algo.detection.heads import YOLOHead
from otx.algo.detection.losses.yolov9_loss import BCELoss, BoxLoss, DFLoss, YOLOv9Criterion
from otx.algo.detection.necks import YOLONeck
from otx.algo.detection.utils.yolov7_v9_utils import Vec2Box
from otx.core.config.data import TileConfig
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.metrics.fmeasure import MeanAveragePrecisionFMeasureCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.detection import ExplainableOTXDetModel

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from typing_extensions import Self

    from otx.core.metrics import MetricCallable
    from otx.core.schedulers import LRSchedulerListCallable
    from otx.core.types.label import LabelInfoTypes


PRETRAINED_WEIGHTS: dict[str, str] = {
    # "yolov9_t": "https://github.com/WongKinYiu/YOLO/releases/download/v1.0-alpha/v9-t.pt", # not supported yet
    "yolov9_s": "https://github.com/WongKinYiu/YOLO/releases/download/v1.0-alpha/v9-s.pt",
    "yolov9_m": "https://github.com/WongKinYiu/YOLO/releases/download/v1.0-alpha/v9-m.pt",
    "yolov9_c": "https://github.com/WongKinYiu/YOLO/releases/download/v1.0-alpha/v9-c.pt",
    # "yolov9_e": "https://github.com/WongKinYiu/YOLO/releases/download/v1.0-alpha/v9-e.pt", # not supported yet
}


def _load_from_state_dict_for_yolov9(
    self: SingleStageDetector,
    state_dict: dict,
    prefix: str,
    local_metadata: dict,
    strict: bool,
    missing_keys: list[str] | str,
    unexpected_keys: list[str] | str,
    error_msgs: list[str] | str,
) -> None:
    backbone_len: int = len(self.backbone.module)
    neck_len: int = len(self.neck.module)  # type: ignore[union-attr]
    new_state_dict_keys: dict[str, str] = {}
    for key in state_dict:
        match = re.match(r"^(\d+)\.(.*)$", key)
        if match:
            orig_idx = int(match.group(1))
            rest_key = match.group(2)

            if orig_idx < backbone_len:
                new_key = f"backbone.module.{orig_idx}.{rest_key}"
            elif orig_idx < backbone_len + neck_len:
                new_idx = orig_idx - backbone_len
                new_key = f"neck.module.{new_idx}.{rest_key}"
            else:  # for bbox_head
                new_idx = orig_idx - backbone_len - neck_len
                new_key = f"bbox_head.module.{new_idx}.{rest_key}"
            new_state_dict_keys[key] = new_key

    for old_key, new_key in new_state_dict_keys.items():
        value = state_dict.pop(old_key)
        state_dict[new_key] = value

    super(SingleStageDetector, self)._load_from_state_dict(
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    )


class YOLOv9(ExplainableOTXDetModel):
    """OTX Detection model class for YOLOv9.

    Default input size per model:
        - yolov9_t : (640, 640) # not supported yet
        - yolov9_s : (640, 640)
        - yolov9_m : (640, 640)
        - yolov9_c : (640, 640)
        - yolov9_e : (640, 640) # not supported yet
    """

    input_size_multiplier = 32  # TODO (sungchul): need to check
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: tuple[float, float, float] = (255.0, 255.0, 255.0)

    def __init__(
        self,
        model_name: Literal["yolov9_t", "yolov9_s", "yolov9_m", "yolov9_c", "yolov9_e"],
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
        backbone = GELAN(model_name=self.model_name)
        neck = YOLONeck(model_name=self.model_name)
        bbox_head = YOLOHead(model_name=self.model_name, num_classes=num_classes)

        # patch _load_from_state_dict
        SingleStageDetector._load_from_state_dict = _load_from_state_dict_for_yolov9  # type: ignore[method-assign] # noqa: SLF001
        detector = SingleStageDetector(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            criterion=None,
        )

        # set criterion
        strides: list[int] | None = [8, 16, 32] if self.model_name == "yolov9_c" else None
        self.vec2box = Vec2Box(detector, self.input_size, strides)
        detector.bbox_head.vec2box = self.vec2box
        detector.criterion = YOLOv9Criterion(
            num_classes=num_classes,
            loss_cls=BCELoss(),
            loss_dfl=DFLoss(self.vec2box),
            loss_iou=BoxLoss(),
            vec2box=self.vec2box,
        )

        return detector

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

    def to(self, *args, **kwargs) -> Self:
        """Sync device of the model and its components."""
        ret = super().to(*args, **kwargs)
        ret.vec2box.update(self.input_size, *args, **kwargs)
        ret.model.criterion.vec2box.update(self.input_size, *args, **kwargs)
        ret.model.criterion.matcher.anchors = ret.model.criterion.matcher.anchors.to(*args, **kwargs)
        ret.model.criterion.loss_dfl.anchors_norm = ret.model.criterion.loss_dfl.anchors_norm.to(*args, **kwargs)
        return ret
