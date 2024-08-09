# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""ATSS model implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from otx.algo.common.backbones import ResNeXt, build_model_including_pytorchcv
from otx.algo.common.losses import CrossEntropyLoss, CrossSigmoidFocalLoss, GIoULoss
from otx.algo.common.utils.coders import DeltaXYWHBBoxCoder
from otx.algo.common.utils.prior_generators import AnchorGenerator
from otx.algo.common.utils.samplers import PseudoSampler
from otx.algo.detection.base_models import SingleStageDetector
from otx.algo.detection.heads import ATSSHead
from otx.algo.detection.necks import FPN
from otx.algo.detection.utils.assigners import ATSSAssigner
from otx.algo.utils.support_otx_v1 import OTXv1Helper
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


class ATSS(ExplainableOTXDetModel):
    """OTX Detection model class for ATSS."""

    def __init__(
        self,
        label_info: LabelInfoTypes,
        input_size: tuple[int, int] = (800, 992),
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

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        if self.input_size is None:
            msg = f"Input size attribute is not set for {self.__class__}"
            raise ValueError(msg)

        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=(1, 3, *self.input_size),
            mean=self.mean,
            std=self.std,
            resize_mode="standard",
            pad_value=0,
            swap_rgb=False,
            via_onnx=True,  # Currently ATSS should be exported through ONNX
            onnx_export_configuration={
                "input_names": ["image"],
                "output_names": ["boxes", "labels"],
                "dynamic_axes": {
                    "image": {0: "batch", 2: "height", 3: "width"},
                    "boxes": {0: "batch", 1: "num_dets"},
                    "labels": {0: "batch", 1: "num_dets"},
                },
                "autograd_inlining": False,
            },
            output_names=["bboxes", "labels", "feature_vector", "saliency_map"] if self.explain_mode else None,
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_det_ckpt(state_dict, add_prefix)


class MobileNetV2ATSS(ATSS):
    """ATSS detector with MobileNetV2 backbone."""

    load_from = (
        "https://storage.openvinotoolkit.org/repositories/"
        "openvino_training_extensions/models/object_detection/v2/mobilenet_v2-atss.pth"
    )
    mean = (0.0, 0.0, 0.0)
    std = (255.0, 255.0, 255.0)

    def _build_model(self, num_classes: int) -> SingleStageDetector:
        train_cfg = {
            "assigner": ATSSAssigner(topk=9),
            "sampler": PseudoSampler(),
            "allowed_border": -1,
            "pos_weight": -1,
            "debug": False,
        }
        test_cfg = {
            "nms": {"type": "nms", "iou_threshold": 0.6},
            "min_bbox_size": 0,
            "score_thr": 0.05,
            "max_per_img": 100,
            "nms_pre": 1000,
        }
        backbone = build_model_including_pytorchcv(
            cfg={
                "type": "mobilenetv2_w1",
                "out_indices": [2, 3, 4, 5],
                "frozen_stages": -1,
                "norm_eval": False,
                "pretrained": True,
            },
        )
        neck = FPN(
            in_channels=[24, 32, 96, 320],
            out_channels=64,
            num_outs=5,
            start_level=1,
            add_extra_convs="on_output",
            relu_before_extra_convs=True,
        )
        bbox_head = ATSSHead(
            num_classes=num_classes,
            in_channels=64,
            anchor_generator=AnchorGenerator(
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[8, 16, 32, 64, 128],
            ),
            bbox_coder=DeltaXYWHBBoxCoder(
                target_means=(0.0, 0.0, 0.0, 0.0),
                target_stds=(0.1, 0.1, 0.2, 0.2),
            ),
            loss_cls=CrossSigmoidFocalLoss(
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0,
            ),
            loss_bbox=GIoULoss(loss_weight=2.0),
            loss_centerness=CrossEntropyLoss(use_sigmoid=True, loss_weight=1.0),
            feat_channels=64,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        return SingleStageDetector(backbone, bbox_head, neck=neck, train_cfg=train_cfg, test_cfg=test_cfg)


class ResNeXt101ATSS(ATSS):
    """ATSS with ResNeXt101 backbone."""

    load_from = (
        "https://storage.openvinotoolkit.org/repositories/"
        "openvino_training_extensions/models/object_detection/v2/resnext101_atss_070623.pth"
    )
    mean = (0.0, 0.0, 0.0)
    std = (255.0, 255.0, 255.0)

    def _build_model(self, num_classes: int) -> SingleStageDetector:
        train_cfg = {
            "assigner": ATSSAssigner(topk=9),
            "sampler": PseudoSampler(),
            "allowed_border": -1,
            "pos_weight": -1,
            "debug": False,
        }
        test_cfg = {
            "nms": {"type": "nms", "iou_threshold": 0.6},
            "min_bbox_size": 0,
            "score_thr": 0.05,
            "max_per_img": 100,
            "nms_pre": 1000,
        }
        backbone = ResNeXt(
            depth=101,
            groups=64,
            frozen_stages=1,
            init_cfg={"type": "Pretrained", "checkpoint": "open-mmlab://resnext101_64x4d"},
        )
        neck = FPN(
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=1,
            add_extra_convs="on_output",
            num_outs=5,
            relu_before_extra_convs=True,
        )
        bbox_head = ATSSHead(
            anchor_generator=AnchorGenerator(
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[8, 16, 32, 64, 128],
            ),
            bbox_coder=DeltaXYWHBBoxCoder(
                target_means=(0.0, 0.0, 0.0, 0.0),
                target_stds=(0.1, 0.1, 0.2, 0.2),
            ),
            loss_cls=CrossSigmoidFocalLoss(
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0,
            ),
            loss_bbox=GIoULoss(loss_weight=2.0),
            loss_centerness=CrossEntropyLoss(use_sigmoid=True, loss_weight=1.0),
            num_classes=num_classes,
            in_channels=256,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        return SingleStageDetector(backbone, bbox_head, neck=neck, train_cfg=train_cfg, test_cfg=test_cfg)

    def to(self, *args, **kwargs) -> Self:
        """Return a model with specified device."""
        ret = super().to(*args, **kwargs)
        if self.device.type == "xpu":
            msg = f"{type(self).__name__} doesn't support XPU."
            raise RuntimeError(msg)
        return ret
