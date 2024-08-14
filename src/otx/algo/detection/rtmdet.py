# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""RTMDet model implementations."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from torch import nn

from otx.algo.common.backbones import CSPNeXt
from otx.algo.common.losses import GIoULoss, QualityFocalLoss
from otx.algo.common.losses.cross_entropy_loss import CrossEntropyLoss
from otx.algo.common.utils.assigners import DynamicSoftLabelAssigner
from otx.algo.common.utils.coders import DistancePointBBoxCoder
from otx.algo.common.utils.prior_generators import MlvlPointGenerator
from otx.algo.common.utils.samplers import PseudoSampler
from otx.algo.detection.base_models import SingleStageDetector
from otx.algo.detection.heads import RTMDetSepBNHead
from otx.algo.detection.necks import CSPNeXtPAFPN
from otx.core.config.data import TileConfig
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.metrics.fmeasure import MeanAveragePrecisionFMeasureCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.detection import ExplainableOTXDetModel
from otx.core.types.export import TaskLevelExportParameters

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallable
    from otx.core.schedulers import LRSchedulerListCallable
    from otx.core.types.label import LabelInfoTypes


class RTMDet(ExplainableOTXDetModel):
    """OTX Detection model class for RTMDet."""

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
            resize_mode="fit_to_window_letterbox",
            pad_value=114,
            swap_rgb=True,
            via_onnx=True,
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

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Defines parameters required to export a particular model implementation."""
        return super()._export_parameters.wrap(optimization_config={"preset": "mixed"})


class RTMDetTiny(RTMDet):
    """RTMDet Tiny Model."""

    load_from = "https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/object_detection/v2/rtmdet_tiny.pth"
    mean = (103.53, 116.28, 123.675)
    std = (57.375, 57.12, 58.395)

    def _build_model(self, num_classes: int) -> RTMDet:
        train_cfg = {
            "assigner": DynamicSoftLabelAssigner(topk=13),
            "sampler": PseudoSampler(),
            "allowed_border": -1,
            "pos_weight": -1,
            "debug": False,
        }

        test_cfg = {
            "nms": {"type": "nms", "iou_threshold": 0.65},
            "score_thr": 0.001,
            "mask_thr_binary": 0.5,
            "max_per_img": 300,
            "min_bbox_size": 0,
            "nms_pre": 30000,
        }

        backbone = CSPNeXt(
            deepen_factor=0.167,
            widen_factor=0.375,
            normalization=nn.BatchNorm2d,
            activation=partial(nn.SiLU, inplace=True),
        )

        neck = CSPNeXtPAFPN(
            in_channels=(96, 192, 384),
            out_channels=96,
            num_csp_blocks=1,
            normalization=nn.BatchNorm2d,
            activation=partial(nn.SiLU, inplace=True),
        )

        bbox_head = RTMDetSepBNHead(
            num_classes=num_classes,
            in_channels=96,
            stacked_convs=2,
            feat_channels=96,
            with_objectness=False,
            anchor_generator=MlvlPointGenerator(offset=0, strides=[8, 16, 32]),
            bbox_coder=DistancePointBBoxCoder(),
            loss_cls=QualityFocalLoss(use_sigmoid=True, beta=2.0, loss_weight=1.0),
            loss_bbox=GIoULoss(loss_weight=2.0),
            loss_centerness=CrossEntropyLoss(use_sigmoid=True, loss_weight=1.0),
            normalization=nn.BatchNorm2d,
            activation=partial(nn.SiLU, inplace=True),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )

        return SingleStageDetector(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
