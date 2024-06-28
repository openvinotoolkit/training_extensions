# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""RTMDet model implementations."""

from __future__ import annotations

from otx.algo.common.backbones import CSPNeXt
from otx.algo.common.losses import GIoULoss, QualityFocalLoss
from otx.algo.common.losses.cross_entropy_loss import CrossEntropyLoss
from otx.algo.common.utils.assigners import DynamicSoftLabelAssigner
from otx.algo.common.utils.coders import DistancePointBBoxCoder
from otx.algo.common.utils.prior_generators import MlvlPointGenerator
from otx.algo.common.utils.samplers import PseudoSampler
from otx.algo.detection.heads import RTMDetSepBNHead
from otx.algo.detection.necks import CSPNeXtPAFPN
from otx.algo.detection.ssd import SingleStageDetector
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.model.detection import ExplainableOTXDetModel
from otx.core.types.export import TaskLevelExportParameters


class RTMDet(ExplainableOTXDetModel):
    """OTX Detection model class for RTMDet."""

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        if self.image_size is None:
            raise ValueError(self.image_size)

        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=self.image_size,
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
    image_size = (1, 3, 640, 640)
    tile_image_size = (1, 3, 640, 640)
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
            norm_cfg={"type": "BN"},
            act_cfg={"type": "SiLU", "inplace": True},
        )

        neck = CSPNeXtPAFPN(
            in_channels=(96, 192, 384),
            out_channels=96,
            num_csp_blocks=1,
            norm_cfg={"type": "BN"},
            act_cfg={"type": "SiLU", "inplace": True},
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
            norm_cfg={"type": "BN"},
            act_cfg={"type": "SiLU", "inplace": True},
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
