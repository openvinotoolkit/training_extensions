# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""MaskRCNN model implementations."""

from __future__ import annotations

from typing import Any, ClassVar

from torchvision.ops import RoIAlign

from otx.algo.common.losses import CrossEntropyLoss, CrossSigmoidFocalLoss, L1Loss
from otx.algo.common.utils.assigners import MaxIoUAssigner
from otx.algo.common.utils.coders import DeltaXYWHBBoxCoder
from otx.algo.common.utils.prior_generators import AnchorGenerator
from otx.algo.common.utils.samplers import RandomSampler
from otx.algo.instance_segmentation.backbones import MaskRCNNBackbone
from otx.algo.instance_segmentation.heads import CustomConvFCBBoxHead, CustomRoIHead, FCNMaskHead, RPNHead
from otx.algo.instance_segmentation.necks import FPN
from otx.algo.instance_segmentation.segmentors.two_stage import TwoStageDetector
from otx.algo.instance_segmentation.utils.roi_extractors import SingleRoIExtractor
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.model.instance_segmentation import ExplainableOTXInstanceSegModel


class MaskRCNN(ExplainableOTXInstanceSegModel):
    """MaskRCNN Model."""

    AVAILABLE_MODEL_VERSIONS: ClassVar[list[str]] = [
        "maskrcnn_resnet_50",
        "maskrcnn_efficientnet_b2b",
        "maskrcnn_swin_tiny",
    ]

    load_from: ClassVar[dict[str, Any]] = {
        "maskrcnn_resnet_50": "https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_mstrain-poly_3x_coco/"
        "mask_rcnn_r50_fpn_mstrain-poly_3x_coco_20210524_201154-21b550bb.pth",
        "maskrcnn_efficientnet_b2b": "https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/"
        "models/instance_segmentation/v2/efficientnet_b2b-mask_rcnn-576x576.pth",
        "maskrcnn_swin_tiny": "https://download.openmmlab.com/mmdetection/v2.0/swin/"
        "mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco/"
        "mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_20210908_165006-90a4008c.pth",
    }

    mean = (123.675, 116.28, 103.53)
    std = (58.395, 57.12, 57.375)
    effnet_std = (1.0, 1.0, 1.0)

    def _build_model(self, num_classes: int) -> TwoStageDetector:
        rpn_assigner = MaxIoUAssigner(
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1,
            match_low_quality=True,
        )

        rpn_sampler = RandomSampler(
            add_gt_as_proposals=False,
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
        )

        rcnn_assigner = MaxIoUAssigner(
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1,
            match_low_quality=True,
        )

        rcnn_sampler = (
            RandomSampler(
                add_gt_as_proposals=True,
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
            ),
        )

        backbone = MaskRCNNBackbone(model_name=self.model_name)
        loss_bbox = L1Loss(loss_weight=1.0)
        loss_rpn_cls = CrossEntropyLoss(loss_weight=1.0, use_sigmoid=True)
        # TODO(someone): performance of CrossSigmoidFocalLoss is worse without mmcv
        # https://github.com/openvinotoolkit/training_extensions/pull/3431
        loss_roi_cls = CrossSigmoidFocalLoss(loss_weight=1.0, use_sigmoid=False)

        neck = FPN(model_name=self.model_name)

        rpn_head = RPNHead(
            anchor_generator=AnchorGenerator(
                strides=[4, 8, 16, 32, 64],
                ratios=[0.5, 1.0, 2.0],
                scales=[8],
            ),
            bbox_coder=DeltaXYWHBBoxCoder(
                target_means=(0.0, 0.0, 0.0, 0.0),
                target_stds=(1.0, 1.0, 1.0, 1.0),
            ),
            assigner=rpn_assigner,
            sampler=rpn_sampler,
            loss_cls=loss_rpn_cls,
            loss_bbox=loss_bbox,
        )

        bbox_head = (
            CustomConvFCBBoxHead(
                num_classes=num_classes,
                reg_class_agnostic=False,
                roi_feat_size=7,
                fc_out_channels=1024,
                in_channels=rpn_head.feat_channels,
                bbox_coder=DeltaXYWHBBoxCoder(
                    target_means=(0.0, 0.0, 0.0, 0.0),
                    target_stds=(0.1, 0.1, 0.2, 0.2),
                ),
                loss_bbox=loss_bbox,
                loss_cls=loss_roi_cls,
            ),
        )

        bbox_roi_extractor = (
            SingleRoIExtractor(
                featmap_strides=[4, 8, 16, 32],
                out_channels=rpn_head.feat_channels,
                roi_layer=RoIAlign(
                    output_size=7,
                    sampling_ratio=0,
                    aligned=True,
                    spatial_scale=1.0,
                ),
            ),
        )

        mask_roi_extractor = (
            SingleRoIExtractor(
                featmap_strides=[4, 8, 16, 32],
                out_channels=rpn_head.feat_channels,
                roi_layer=RoIAlign(
                    output_size=14,
                    sampling_ratio=0,
                    aligned=True,
                    spatial_scale=1.0,
                ),
            ),
        )

        mask_head = (
            FCNMaskHead(
                conv_out_channels=rpn_head.feat_channels,
                in_channels=rpn_head.feat_channels,
                loss_mask=CrossEntropyLoss(loss_weight=1.0, use_mask=True),
                num_classes=num_classes,
                num_convs=4,
            ),
        )

        roi_head = CustomRoIHead(
            # TODO test_cfg consist of nms cfg, make it available there
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            assigner=rcnn_assigner,
            sampler=rcnn_sampler,
        )

        # TODO check that other model are aligned here

        return TwoStageDetector(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
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
            resize_mode="fit_to_window",
            pad_value=0,
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
            output_names=["bboxes", "labels", "masks"],
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_iseg_ckpt(state_dict, add_prefix)

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for MaskRCNN-Eff."""
        if self.model_name == "maskrcnn_efficientnet_b2b":
            return {
                "ignored_scope": {
                    "types": ["Add", "Divide", "Multiply", "Sigmoid"],
                    "validate": False,
                },
                "preset": "mixed",
            }

        elif self.model_name == "maskrcnn_swin_t":
            return {
                "ignored_scope": {
                    "types": [
                        "Add",
                        "MVN",
                        "Divide",
                        "Multiply",
                    ],
                    "validate": False,
                },
                "preset": "mixed",
            }

        return {}
