# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""RTMDetInst model implementations."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from torch import nn

from otx.algo.common.backbones import CSPNeXt
from otx.algo.common.losses import CrossEntropyLoss, GIoULoss, QualityFocalLoss
from otx.algo.common.utils.assigners import DynamicSoftLabelAssigner
from otx.algo.common.utils.coders import DistancePointBBoxCoder
from otx.algo.common.utils.prior_generators import MlvlPointGenerator
from otx.algo.common.utils.samplers import PseudoSampler
from otx.algo.detection.base_models import SingleStageDetector
from otx.algo.detection.necks import CSPNeXtPAFPN
from otx.algo.instance_segmentation.heads import RTMDetInsSepBNHead
from otx.algo.instance_segmentation.losses import DiceLoss
from otx.algo.modules.norm import build_norm_layer
from otx.core.config.data import TileConfig
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.metrics.mean_ap import MaskRLEMeanAPFMeasureCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.instance_segmentation import ExplainableOTXInstanceSegModel

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from torch import Tensor

    from otx.core.metrics import MetricCallable
    from otx.core.schedulers import LRSchedulerListCallable
    from otx.core.types.label import LabelInfoTypes


class RTMDetInst(ExplainableOTXInstanceSegModel):
    """Implementation of RTMDet for instance segmentation."""

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
            swap_rgb=False,
            via_onnx=True,
            onnx_export_configuration={
                "input_names": ["image"],
                "output_names": ["boxes", "labels", "masks"],
                "dynamic_axes": {
                    "image": {0: "batch", 2: "height", 3: "width"},
                    "boxes": {0: "batch", 1: "num_dets"},
                    "labels": {0: "batch", 1: "num_dets"},
                    "masks": {0: "batch", 1: "num_dets", 2: "height", 3: "width"},
                },
                "opset_version": 11,
                "autograd_inlining": False,
            },
            output_names=["bboxes", "labels", "masks", "feature_vector", "saliency_map"] if self.explain_mode else None,
        )

    def forward_for_tracing(self, inputs: Tensor) -> tuple[Tensor, ...]:
        """Forward function for export.

        NOTE : RTMDetInst uses explain_mode unlike other models.
        """
        shape = (int(inputs.shape[2]), int(inputs.shape[3]))
        meta_info = {
            "pad_shape": shape,
            "batch_input_shape": shape,
            "img_shape": shape,
            "scale_factor": (1.0, 1.0),
        }
        meta_info_list = [meta_info] * len(inputs)
        return self.model.export(inputs, meta_info_list, explain_mode=self.explain_mode)


class RTMDetInstTiny(RTMDetInst):
    """RTMDetInst Tiny Model."""

    load_from = (
        "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco/"
        "rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth"
    )
    mean = (123.675, 116.28, 103.53)
    std = (58.395, 57.12, 57.375)

    def __init__(
        self,
        label_info: LabelInfoTypes,
        input_size: tuple[int, int] = (640, 640),
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MaskRLEMeanAPFMeasureCallable,
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
        train_cfg = {
            "assigner": DynamicSoftLabelAssigner(topk=13),
            "sampler": PseudoSampler(),
            "allowed_border": -1,
            "pos_weight": -1,
            "debug": False,
        }

        test_cfg = {
            "nms": {"type": "nms", "iou_threshold": 0.5},
            "score_thr": 0.05,
            "mask_thr_binary": 0.5,
            "max_per_img": 100,
            "min_bbox_size": 0,
            "nms_pre": 300,
        }

        backbone = CSPNeXt(
            arch="P5",
            expand_ratio=0.5,
            deepen_factor=0.167,
            widen_factor=0.375,
            channel_attention=True,
            normalization=nn.BatchNorm2d,
            activation=partial(nn.SiLU, inplace=True),
        )

        neck = CSPNeXtPAFPN(
            in_channels=(96, 192, 384),
            out_channels=96,
            num_csp_blocks=1,
            expand_ratio=0.5,
            normalization=nn.BatchNorm2d,
            activation=partial(nn.SiLU, inplace=True),
        )

        bbox_head = RTMDetInsSepBNHead(
            num_classes=num_classes,
            in_channels=96,
            stacked_convs=2,
            share_conv=True,
            pred_kernel_size=1,
            feat_channels=96,
            normalization=partial(build_norm_layer, nn.BatchNorm2d, requires_grad=True),
            activation=partial(nn.SiLU, inplace=True),
            anchor_generator=MlvlPointGenerator(
                offset=0,
                strides=[8, 16, 32],
            ),
            bbox_coder=DistancePointBBoxCoder(),
            loss_centerness=CrossEntropyLoss(use_sigmoid=True, loss_weight=1.0),
            loss_cls=QualityFocalLoss(
                use_sigmoid=True,
                beta=2.0,
                loss_weight=1.0,
            ),
            loss_bbox=GIoULoss(loss_weight=2.0),
            loss_mask=DiceLoss(
                loss_weight=2.0,
                eps=5.0e-06,
                reduction="mean",
            ),
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
