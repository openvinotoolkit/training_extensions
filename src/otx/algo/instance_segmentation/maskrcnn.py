# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""MaskRCNN model implementations."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from mmcv.ops import RoIAlign
from mmengine.structures import InstanceData
from omegaconf import DictConfig

from otx.algo.detection.backbones.pytorchcv_backbones import _build_pytorchcv_model
from otx.algo.detection.heads.custom_anchor_generator import AnchorGenerator
from otx.algo.detection.heads.delta_xywh_bbox_coder import DeltaXYWHBBoxCoder
from otx.algo.detection.losses.cross_entropy_loss import CrossEntropyLoss
from otx.algo.detection.losses.cross_focal_loss import CrossSigmoidFocalLoss
from otx.algo.detection.losses.smooth_l1_loss import L1Loss
from otx.algo.instance_segmentation.mmdet.models.backbones import ResNet
from otx.algo.instance_segmentation.mmdet.models.custom_roi_head import CustomConvFCBBoxHead, CustomRoIHead
from otx.algo.instance_segmentation.mmdet.models.dense_heads import RPNHead
from otx.algo.instance_segmentation.mmdet.models.detectors import MaskRCNN
from otx.algo.instance_segmentation.mmdet.models.mask_heads import FCNMaskHead
from otx.algo.instance_segmentation.mmdet.models.necks import FPN
from otx.algo.instance_segmentation.mmdet.models.roi_extractors import SingleRoIExtractor
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.config.data import TileConfig
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.metrics.mean_ap import MaskRLEMeanAPCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.instance_segmentation import MMDetInstanceSegCompatibleModel
from otx.core.model.utils.mmdet import DetDataPreprocessor
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.label import LabelInfoTypes
from otx.core.utils.build import modify_num_classes
from otx.core.utils.config import convert_conf_to_mmconfig_dict
from otx.core.utils.utils import get_mean_std_from_data_processing

if TYPE_CHECKING:
    import torch
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from torch.nn.modules import Module

    from otx.core.metrics import MetricCallable


class MMDetMaskRCNN(MMDetInstanceSegCompatibleModel):
    """MaskRCNN Model."""

    def __init__(
        self,
        label_info: LabelInfoTypes,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MaskRLEMeanAPCallable,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
    ) -> None:
        super().__init__(
            label_info=label_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            tile_config=tile_config,
        )
        self.image_size = (1, 3, 1024, 1024)
        self.tile_image_size = (1, 3, 512, 512)

    def get_classification_layers(self, prefix: str = "") -> dict[str, dict[str, int]]:
        """Return classification layer names by comparing two different number of classes models.

        Args:
            config (DictConfig): Config for building model.
            model_registry (Registry): Registry for building model.
            prefix (str): Prefix of model param name.
                Normally it is "model." since OTXModel set it's nn.Module model as self.model

        Return:
            dict[str, dict[str, int]]
            A dictionary contain classification layer's name and information.
            Stride means dimension of each classes, normally stride is 1, but sometimes it can be 4
            if the layer is related bbox regression for object detection.
            Extra classes is default class except class from data.
            Normally it is related with background classes.
        """
        sample_model_dict = self._build_model(num_classes=5).state_dict()
        incremental_model_dict = self._build_model(num_classes=6).state_dict()

        classification_layers = {}
        for key in sample_model_dict:
            if sample_model_dict[key].shape != incremental_model_dict[key].shape:
                sample_model_dim = sample_model_dict[key].shape[0]
                incremental_model_dim = incremental_model_dict[key].shape[0]
                stride = incremental_model_dim - sample_model_dim
                num_extra_classes = 6 * sample_model_dim - 5 * incremental_model_dim
                classification_layers[prefix + key] = {"stride": stride, "num_extra_classes": num_extra_classes}
        return classification_layers

    def _create_model(self) -> Module:
        from mmengine.runner import load_checkpoint

        detector = self._build_model(num_classes=self.label_info.num_classes)
        self.classification_layers = self.get_classification_layers("model.")

        if self.load_from is not None:
            load_checkpoint(detector, self.load_from, map_location="cpu")
        return detector

    def _build_model(self, num_classes: int) -> MMDetMaskRCNN:
        raise NotImplementedError

    @property
    def _exporter(self) -> OTXModelExporter:
        raise NotImplementedError

    def forward_for_tracing(
        self,
        inputs: torch.Tensor,
    ) -> list[InstanceData]:
        """Forward function for export."""
        shape = (int(inputs.shape[2]), int(inputs.shape[3]))
        meta_info = {
            "pad_shape": shape,
            "batch_input_shape": shape,
            "img_shape": shape,
            "scale_factor": (1.0, 1.0),
        }
        sample = InstanceData(
            metainfo=meta_info,
        )
        data_samples = [sample] * len(inputs)
        return self.model.export(
            inputs,
            data_samples,
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_iseg_ckpt(state_dict, add_prefix)


class MaskRCNNResNet50(MMDetMaskRCNN):
    load_from = (
        "https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_mstrain-poly_3x_coco/"
        "mask_rcnn_r50_fpn_mstrain-poly_3x_coco_20210524_201154-21b550bb.pth"
    )

    def __init__(
        self,
        label_info: LabelInfoTypes,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MaskRLEMeanAPCallable,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
    ) -> None:
        super().__init__(label_info, optimizer, scheduler, metric, torch_compile, tile_config)

    def _build_model(self, num_classes: int) -> MMDetMaskRCNN:
        train_cfg = DictConfig(
            {
                "rpn": {
                    "allowed_border": -1,
                    "debug": False,
                    "pos_weight": -1,
                    "assigner": {
                        "type": "MaxIoUAssigner",
                        "ignore_iof_thr": -1,
                        "match_low_quality": True,
                        "pos_iou_thr": 0.7,
                        "neg_iou_thr": 0.3,
                        "min_pos_iou": 0.3,
                    },
                    "sampler": {
                        "type": "RandomSampler",
                        "add_gt_as_proposals": False,
                        "neg_pos_ub": -1,
                        "num": 256,
                        "pos_fraction": 0.5,
                    },
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
                    "assigner": {
                        "type": "MaxIoUAssigner",
                        "ignore_iof_thr": -1,
                        "match_low_quality": True,
                        "pos_iou_thr": 0.5,
                        "neg_iou_thr": 0.5,
                        "min_pos_iou": 0.5,
                    },
                    "sampler": {
                        "type": "RandomSampler",
                        "add_gt_as_proposals": True,
                        "neg_pos_ub": -1,
                        "num": 512,
                        "pos_fraction": 0.25,
                    },
                    "debug": False,
                    "mask_size": 28,
                    "pos_weight": -1,
                },
            },
        )
        test_cfg = DictConfig(
            {
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
            },
        )

        data_preprocessor = DetDataPreprocessor(
            bgr_to_rgb=False,
            mean=[123.675, 116.28, 103.53],
            pad_mask=True,
            pad_size_divisor=32,
            std=[58.395, 57.12, 57.375],
            non_blocking=True,
        )

        backbone = ResNet(
            depth=50,
            frozen_stages=1,
            norm_cfg={"type": "BN", "requires_grad": True},
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
            train_cfg=train_cfg.rpn,
            test_cfg=test_cfg.rpn,
        )

        roi_head = CustomRoIHead(
            bbox_roi_extractor=SingleRoIExtractor(
                featmap_strides=[4, 8, 16, 32],
                out_channels=256,
                roi_layer=RoIAlign(output_size=7, sampling_ratio=0),
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
                loss_cls=CrossSigmoidFocalLoss(loss_weight=1.0, use_sigmoid=False),
            ),
            mask_roi_extractor=SingleRoIExtractor(
                featmap_strides=[4, 8, 16, 32],
                out_channels=256,
                roi_layer=RoIAlign(output_size=14, sampling_ratio=0),
            ),
            mask_head=FCNMaskHead(
                conv_out_channels=256,
                in_channels=256,
                loss_mask=CrossEntropyLoss(loss_weight=1.0, use_mask=True),
                num_classes=num_classes,
                num_convs=4,
            ),
            train_cfg=train_cfg.rcnn,
            test_cfg=test_cfg.rcnn,
        )

        return MaskRCNN(
            data_preprocessor=data_preprocessor,
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        if self.image_size is None:
            raise ValueError(self.image_size)

        mean = (123.675, 116.28, 103.53)
        std = (58.395, 57.12, 57.375)

        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=self.image_size,
            mean=mean,
            std=std,
            resize_mode="standard",
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


class MaskRCNNEfficientNet(MMDetMaskRCNN):
    load_from = ()

    def __init__(
        self,
        label_info: LabelInfoTypes,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MaskRLEMeanAPCallable,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
    ) -> None:
        super().__init__(label_info, optimizer, scheduler, metric, torch_compile, tile_config)

    def _build_model(self, num_classes: int) -> MMDetMaskRCNN:
        backbone = _build_pytorchcv_model("efficientnet_b2b", **backbone)
        return MaskRCNN

    @property
    def _exporter(self) -> OTXModelExporter:
        raise NotImplementedError


class MaskRCNNSwinT(MMDetInstanceSegCompatibleModel):
    """MaskRCNNSwinT Model."""

    def __init__(
        self,
        label_info: LabelInfoTypes,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MaskRLEMeanAPCallable,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
    ) -> None:
        super().__init__(
            label_info=label_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            tile_config=tile_config,
        )
        self.image_size = (1, 3, 1344, 1344)
        self.tile_image_size = (1, 3, 512, 512)

    def get_classification_layers(self, config: DictConfig, prefix: str = "") -> dict[str, dict[str, int]]:
        """Return classification layer names by comparing two different number of classes models.

        Args:
            config (DictConfig): Config for building model.
            model_registry (Registry): Registry for building model.
            prefix (str): Prefix of model param name.
                Normally it is "model." since OTXModel set it's nn.Module model as self.model

        Return:
            dict[str, dict[str, int]]
            A dictionary contain classification layer's name and information.
            Stride means dimension of each classes, normally stride is 1, but sometimes it can be 4
            if the layer is related bbox regression for object detection.
            Extra classes is default class except class from data.
            Normally it is related with background classes.
        """
        sample_config = deepcopy(config)
        modify_num_classes(sample_config, 5)
        sample_model_dict = MaskRCNN(**convert_conf_to_mmconfig_dict(sample_config, to="list")).state_dict()

        modify_num_classes(sample_config, 6)
        incremental_model_dict = MaskRCNN(
            **convert_conf_to_mmconfig_dict(sample_config, to="list"),
        ).state_dict()

        classification_layers = {}
        for key in sample_model_dict:
            if sample_model_dict[key].shape != incremental_model_dict[key].shape:
                sample_model_dim = sample_model_dict[key].shape[0]
                incremental_model_dim = incremental_model_dict[key].shape[0]
                stride = incremental_model_dim - sample_model_dim
                num_extra_classes = 6 * sample_model_dim - 5 * incremental_model_dim
                classification_layers[prefix + key] = {"stride": stride, "num_extra_classes": num_extra_classes}
        return classification_layers

    def _create_model(self) -> Module:
        from mmengine.runner import load_checkpoint

        config = deepcopy(self.config)
        self.classification_layers = self.get_classification_layers(config, "model.")
        detector = MaskRCNN(**convert_conf_to_mmconfig_dict(config, to="list"))
        if self.load_from is not None:
            load_checkpoint(detector, self.load_from, map_location="cpu")
        return detector

    def forward_for_tracing(
        self,
        inputs: torch.Tensor,
    ) -> list[InstanceData]:
        """Forward function for export."""
        shape = (int(inputs.shape[2]), int(inputs.shape[3]))
        meta_info = {
            "pad_shape": shape,
            "batch_input_shape": shape,
            "img_shape": shape,
            "scale_factor": (1.0, 1.0),
        }
        sample = InstanceData(
            metainfo=meta_info,
        )
        data_samples = [sample] * len(inputs)
        return self.model.export(
            inputs,
            data_samples,
        )

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        if self.image_size is None:
            raise ValueError(self.image_size)

        mean, std = get_mean_std_from_data_processing(self.config)

        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=self.image_size,
            mean=mean,
            std=std,
            resize_mode="standard",
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
            output_names=["feature_vector", "saliency_map"] if self.explain_mode else None,
        )
