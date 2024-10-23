# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""MaskRCNN model implementations."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

from torch import nn
from torchvision.ops import RoIAlign

from otx.algo.common.backbones import ResNet, build_model_including_pytorchcv
from otx.algo.common.losses import CrossEntropyLoss, CrossSigmoidFocalLoss, L1Loss
from otx.algo.common.utils.assigners import MaxIoUAssigner
from otx.algo.common.utils.coders import DeltaXYWHBBoxCoder
from otx.algo.common.utils.prior_generators import AnchorGenerator
from otx.algo.common.utils.samplers import RandomSampler
from otx.algo.instance_segmentation.backbones import SwinTransformer
from otx.algo.instance_segmentation.heads import CustomConvFCBBoxHead, CustomRoIHead, FCNMaskHead, RPNHead
from otx.algo.instance_segmentation.necks import FPN
from otx.algo.instance_segmentation.two_stage import TwoStageDetector
from otx.algo.instance_segmentation.utils.roi_extractors import SingleRoIExtractor
from otx.algo.modules.norm import build_norm_layer
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.config.data import TileConfig
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.metrics.mean_ap import MaskRLEMeanAPFMeasureCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.instance_segmentation import ExplainableOTXInstanceSegModel

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallable
    from otx.core.schedulers import LRSchedulerListCallable
    from otx.core.types.label import LabelInfoTypes


class MaskRCNN(ExplainableOTXInstanceSegModel):
    """MaskRCNN Model."""

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
            output_names=["bboxes", "labels", "masks", "feature_vector", "saliency_map"] if self.explain_mode else None,
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_iseg_ckpt(state_dict, add_prefix)


class MaskRCNNResNet50(MaskRCNN):
    """MaskRCNN with ResNet50 backbone."""

    load_from = (
        "https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_mstrain-poly_3x_coco/"
        "mask_rcnn_r50_fpn_mstrain-poly_3x_coco_20210524_201154-21b550bb.pth"
    )
    mean = (123.675, 116.28, 103.53)
    std = (58.395, 57.12, 57.375)

    def __init__(
        self,
        label_info: LabelInfoTypes,
        input_size: tuple[int, int] = (1024, 1024),
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

    def _build_model(self, num_classes: int) -> TwoStageDetector:
        train_cfg = {
            "rpn": {
                "allowed_border": -1,
                "debug": False,
                "pos_weight": -1,
                "assigner": MaxIoUAssigner(
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    ignore_iof_thr=-1,
                    match_low_quality=True,
                ),
                "sampler": RandomSampler(
                    add_gt_as_proposals=False,
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                ),
            },
            "rpn_proposal": {
                "max_per_img": 1000,
                "min_bbox_size": 0,
                "nms": {
                    "type": "nms",
                    "iou_threshold": 0.7,
                },
                "nms_pre": 2000,
            },
            "rcnn": {
                "assigner": MaxIoUAssigner(
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    ignore_iof_thr=-1,
                    match_low_quality=True,
                ),
                "sampler": RandomSampler(
                    add_gt_as_proposals=True,
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                ),
                "debug": False,
                "mask_size": 28,
                "pos_weight": -1,
            },
        }

        test_cfg = {
            "rpn": {
                "max_per_img": 1000,
                "min_bbox_size": 0,
                "nms": {
                    "type": "nms",
                    "iou_threshold": 0.7,
                },
                "nms_pre": 1000,
            },
            "rcnn": {
                "mask_thr_binary": 0.5,
                "max_per_img": 100,
                "nms": {
                    "type": "nms",
                    "iou_threshold": 0.5,
                },
                "score_thr": 0.05,
            },
        }

        backbone = ResNet(
            depth=50,
            frozen_stages=1,
            normalization=partial(build_norm_layer, nn.BatchNorm2d, requires_grad=True),
            norm_eval=True,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
        )

        neck = FPN(
            in_channels=[256, 512, 1024, 2048],
            num_outs=5,
            out_channels=256,
        )

        rpn_head = RPNHead(
            in_channels=256,
            feat_channels=256,
            anchor_generator=AnchorGenerator(
                strides=[4, 8, 16, 32, 64],
                ratios=[0.5, 1.0, 2.0],
                scales=[8],
            ),
            bbox_coder=DeltaXYWHBBoxCoder(
                target_means=(0.0, 0.0, 0.0, 0.0),
                target_stds=(1.0, 1.0, 1.0, 1.0),
            ),
            loss_bbox=L1Loss(loss_weight=1.0),
            loss_cls=CrossEntropyLoss(loss_weight=1.0, use_sigmoid=True),
            train_cfg=train_cfg["rpn"],
            test_cfg=test_cfg["rpn"],
        )

        roi_head = CustomRoIHead(
            bbox_roi_extractor=SingleRoIExtractor(
                featmap_strides=[4, 8, 16, 32],
                out_channels=256,
                roi_layer=RoIAlign(
                    output_size=7,
                    sampling_ratio=0,
                    aligned=True,
                    spatial_scale=1.0,
                ),
            ),
            bbox_head=CustomConvFCBBoxHead(
                num_classes=num_classes,
                reg_class_agnostic=False,
                roi_feat_size=7,
                fc_out_channels=1024,
                in_channels=256,
                bbox_coder=DeltaXYWHBBoxCoder(
                    target_means=(0.0, 0.0, 0.0, 0.0),
                    target_stds=(0.1, 0.1, 0.2, 0.2),
                ),
                loss_bbox=L1Loss(loss_weight=1.0),
                # TODO(someone): performance of CrossSigmoidFocalLoss is worse without mmcv
                # https://github.com/openvinotoolkit/training_extensions/pull/3431
                loss_cls=CrossSigmoidFocalLoss(loss_weight=1.0, use_sigmoid=False),
            ),
            mask_roi_extractor=SingleRoIExtractor(
                featmap_strides=[4, 8, 16, 32],
                out_channels=256,
                roi_layer=RoIAlign(
                    output_size=14,
                    sampling_ratio=0,
                    aligned=True,
                    spatial_scale=1.0,
                ),
            ),
            mask_head=FCNMaskHead(
                conv_out_channels=256,
                in_channels=256,
                loss_mask=CrossEntropyLoss(loss_weight=1.0, use_mask=True),
                num_classes=num_classes,
                num_convs=4,
            ),
            train_cfg=train_cfg["rcnn"],
            test_cfg=test_cfg["rcnn"],
        )

        return TwoStageDetector(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )


class MaskRCNNEfficientNet(MaskRCNN):
    """MaskRCNN with efficientnet_b2b backbone."""

    load_from = (
        "https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/"
        "models/instance_segmentation/v2/efficientnet_b2b-mask_rcnn-576x576.pth"
    )
    mean = (123.675, 116.28, 103.53)
    std = (1.0, 1.0, 1.0)

    def __init__(
        self,
        label_info: LabelInfoTypes,
        input_size: tuple[int, int] = (1024, 1024),
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

    def _build_model(self, num_classes: int) -> TwoStageDetector:
        train_cfg = {
            "rpn": {
                "assigner": MaxIoUAssigner(
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    ignore_iof_thr=-1,
                    match_low_quality=True,
                    gpu_assign_thr=300,
                ),
                "sampler": RandomSampler(
                    add_gt_as_proposals=False,
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                ),
                "allowed_border": -1,
                "debug": False,
                "pos_weight": -1,
            },
            "rpn_proposal": {
                "max_per_img": 1000,
                "min_bbox_size": 0,
                "nms": {
                    "type": "nms",
                    "iou_threshold": 0.8,
                },
                "nms_pre": 2000,
            },
            "rcnn": {
                "assigner": MaxIoUAssigner(
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    ignore_iof_thr=-1,
                    match_low_quality=True,
                    gpu_assign_thr=300,
                ),
                "sampler": RandomSampler(
                    add_gt_as_proposals=True,
                    num=256,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                ),
                "debug": False,
                "mask_size": 28,
                "pos_weight": -1,
            },
        }

        test_cfg = {
            "rpn": {
                "nms_across_levels": False,
                "nms_pre": 800,
                "max_per_img": 500,
                "min_bbox_size": 0,
                "nms": {
                    "type": "nms",
                    "iou_threshold": 0.8,
                },
            },
            "rcnn": {
                "mask_thr_binary": 0.5,
                "max_per_img": 500,
                "nms": {
                    "type": "nms",
                    "iou_threshold": 0.5,
                },
                "score_thr": 0.05,
            },
        }

        backbone = build_model_including_pytorchcv(
            cfg={
                "type": "efficientnet_b2b",
                "out_indices": [2, 3, 4, 5],
                "frozen_stages": -1,
                "pretrained": True,
                "activation": nn.SiLU,
                "normalization": partial(build_norm_layer, nn.BatchNorm2d, requires_grad=True),
            },
        )

        neck = FPN(
            in_channels=[24, 48, 120, 352],
            out_channels=80,
            num_outs=5,
        )

        rpn_head = RPNHead(
            in_channels=80,
            feat_channels=80,
            anchor_generator=AnchorGenerator(
                strides=[4, 8, 16, 32, 64],
                ratios=[0.5, 1.0, 2.0],
                scales=[8],
            ),
            bbox_coder=DeltaXYWHBBoxCoder(
                target_means=(0.0, 0.0, 0.0, 0.0),
                target_stds=(1.0, 1.0, 1.0, 1.0),
            ),
            loss_bbox=L1Loss(loss_weight=1.0),
            loss_cls=CrossEntropyLoss(loss_weight=1.0, use_sigmoid=True),
            train_cfg=train_cfg["rpn"],
            test_cfg=test_cfg["rpn"],
        )

        roi_head = CustomRoIHead(
            bbox_roi_extractor=SingleRoIExtractor(
                featmap_strides=[4, 8, 16, 32],
                out_channels=80,
                roi_layer=RoIAlign(
                    output_size=7,
                    sampling_ratio=0,
                    aligned=True,
                    spatial_scale=1.0,
                ),
            ),
            bbox_head=CustomConvFCBBoxHead(
                num_classes=num_classes,
                reg_class_agnostic=False,
                roi_feat_size=7,
                fc_out_channels=1024,
                in_channels=80,
                bbox_coder=DeltaXYWHBBoxCoder(
                    target_means=(0.0, 0.0, 0.0, 0.0),
                    target_stds=(0.1, 0.1, 0.2, 0.2),
                ),
                loss_bbox=L1Loss(loss_weight=1.0),
                # TODO(someone): performance of CrossSigmoidFocalLoss is worse without mmcv
                # https://github.com/openvinotoolkit/training_extensions/pull/3431
                loss_cls=CrossSigmoidFocalLoss(loss_weight=1.0, use_sigmoid=False),
            ),
            mask_roi_extractor=SingleRoIExtractor(
                featmap_strides=[4, 8, 16, 32],
                out_channels=80,
                roi_layer=RoIAlign(
                    output_size=14,
                    sampling_ratio=0,
                    aligned=True,
                    spatial_scale=1.0,
                ),
            ),
            mask_head=FCNMaskHead(
                conv_out_channels=80,
                in_channels=80,
                loss_mask=CrossEntropyLoss(loss_weight=1.0, use_mask=True),
                num_classes=num_classes,
                num_convs=4,
            ),
            train_cfg=train_cfg["rcnn"],
            test_cfg=test_cfg["rcnn"],
        )

        return TwoStageDetector(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for MaskRCNN-Eff."""
        return {
            "ignored_scope": {
                "types": ["Add", "Divide", "Multiply", "Sigmoid"],
                "validate": False,
            },
            "preset": "mixed",
        }


class MaskRCNNSwinT(MaskRCNN):
    """MaskRCNNSwinT Model."""

    load_from = (
        "https://download.openmmlab.com/mmdetection/v2.0/swin/"
        "mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco/"
        "mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_20210908_165006-90a4008c.pth"
    )
    mean = (123.675, 116.28, 103.53)
    std = (58.395, 57.12, 57.375)

    def __init__(
        self,
        label_info: LabelInfoTypes,
        input_size: tuple[int, int] = (1344, 1344),
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

    def _build_model(self, num_classes: int) -> TwoStageDetector:
        train_cfg = {
            "rpn": {
                "assigner": MaxIoUAssigner(
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    ignore_iof_thr=-1,
                    match_low_quality=True,
                ),
                "sampler": RandomSampler(
                    add_gt_as_proposals=False,
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                ),
                "allowed_border": -1,
                "debug": False,
                "pos_weight": -1,
            },
            "rpn_proposal": {
                "max_per_img": 1000,
                "min_bbox_size": 0,
                "nms": {
                    "type": "nms",
                    "iou_threshold": 0.7,
                },
                "nms_pre": 2000,
            },
            "rcnn": {
                "assigner": MaxIoUAssigner(
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    ignore_iof_thr=-1,
                    match_low_quality=True,
                ),
                "sampler": RandomSampler(
                    add_gt_as_proposals=True,
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                ),
                "debug": False,
                "mask_size": 28,
                "pos_weight": -1,
            },
        }

        test_cfg = {
            "rpn": {
                "max_per_img": 1000,
                "min_bbox_size": 0,
                "nms": {
                    "type": "nms",
                    "iou_threshold": 0.7,
                },
                "nms_pre": 1000,
            },
            "rcnn": {
                "mask_thr_binary": 0.5,
                "max_per_img": 100,
                "nms": {
                    "type": "nms",
                    "iou_threshold": 0.5,
                },
                "score_thr": 0.05,
            },
        }

        backbone = SwinTransformer(
            embed_dims=96,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            window_size=7,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.2,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            with_cp=False,
            convert_weights=True,
        )

        neck = FPN(
            in_channels=[96, 192, 384, 768],
            out_channels=256,
            num_outs=5,
        )

        rpn_head = RPNHead(
            in_channels=256,
            feat_channels=256,
            anchor_generator=AnchorGenerator(
                strides=[4, 8, 16, 32, 64],
                ratios=[0.5, 1.0, 2.0],
                scales=[8],
            ),
            bbox_coder=DeltaXYWHBBoxCoder(
                target_means=(0.0, 0.0, 0.0, 0.0),
                target_stds=(1.0, 1.0, 1.0, 1.0),
            ),
            loss_bbox=L1Loss(loss_weight=1.0),
            loss_cls=CrossEntropyLoss(loss_weight=1.0, use_sigmoid=True),
            train_cfg=train_cfg["rpn"],
            test_cfg=test_cfg["rpn"],
        )

        roi_head = CustomRoIHead(
            bbox_roi_extractor=SingleRoIExtractor(
                featmap_strides=[4, 8, 16, 32],
                out_channels=256,
                roi_layer=RoIAlign(
                    output_size=7,
                    sampling_ratio=0,
                    aligned=True,
                    spatial_scale=1.0,
                ),
            ),
            bbox_head=CustomConvFCBBoxHead(
                num_classes=num_classes,
                reg_class_agnostic=False,
                roi_feat_size=7,
                fc_out_channels=1024,
                in_channels=256,
                bbox_coder=DeltaXYWHBBoxCoder(
                    target_means=(0.0, 0.0, 0.0, 0.0),
                    target_stds=(0.1, 0.1, 0.2, 0.2),
                ),
                loss_bbox=L1Loss(loss_weight=1.0),
                # TODO(someone): performance of CrossSigmoidFocalLoss is worse without mmcv
                # https://github.com/openvinotoolkit/training_extensions/pull/3431
                loss_cls=CrossSigmoidFocalLoss(loss_weight=1.0, use_sigmoid=False),
            ),
            mask_roi_extractor=SingleRoIExtractor(
                featmap_strides=[4, 8, 16, 32],
                out_channels=256,
                roi_layer=RoIAlign(
                    output_size=14,
                    sampling_ratio=0,
                    aligned=True,
                    spatial_scale=1.0,
                ),
            ),
            mask_head=FCNMaskHead(
                conv_out_channels=256,
                in_channels=256,
                loss_mask=CrossEntropyLoss(loss_weight=1.0, use_mask=True),
                num_classes=num_classes,
                num_convs=4,
            ),
            train_cfg=train_cfg["rcnn"],
            test_cfg=test_cfg["rcnn"],
        )

        return TwoStageDetector(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for MaskRCNN-SwinT."""
        return {
            "model_type": "transformer",
            "ignored_scope": {
                "patterns": [".*head.*"],
                "validate": False,
            },
            "advanced_parameters": {
                "smooth_quant_alpha": -1,
                "activations_range_estimator_params": {
                    "min": {
                        "statistics_type": "QUANTILE",
                        "aggregator_type": "MIN",
                        "quantile_outlier_prob": "1e-4",
                    },
                    "max": {
                        "statistics_type": "QUANTILE",
                        "aggregator_type": "MAX",
                        "quantile_outlier_prob": "1e-4",
                    },
                },
            },
        }
