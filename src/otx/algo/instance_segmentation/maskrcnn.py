# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""MaskRCNN model implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from omegaconf import DictConfig
from torchvision import tv_tensors
from torchvision.ops import RoIAlign

from otx.algo.detection.backbones.pytorchcv_backbones import _build_model_including_pytorchcv
from otx.algo.detection.heads.anchor_generator import AnchorGenerator
from otx.algo.detection.heads.base_sampler import RandomSampler
from otx.algo.detection.heads.delta_xywh_bbox_coder import DeltaXYWHBBoxCoder
from otx.algo.detection.heads.max_iou_assigner import MaxIoUAssigner
from otx.algo.detection.losses.cross_entropy_loss import CrossEntropyLoss
from otx.algo.detection.losses.cross_focal_loss import CrossSigmoidFocalLoss
from otx.algo.detection.losses.smooth_l1_loss import L1Loss
from otx.algo.instance_segmentation.mmdet.models.backbones import ResNet, SwinTransformer
from otx.algo.instance_segmentation.mmdet.models.custom_roi_head import CustomConvFCBBoxHead, CustomRoIHead
from otx.algo.instance_segmentation.mmdet.models.dense_heads import RPNHead
from otx.algo.instance_segmentation.mmdet.models.detectors import MaskRCNN
from otx.algo.instance_segmentation.mmdet.models.mask_heads import FCNMaskHead
from otx.algo.instance_segmentation.mmdet.models.necks import FPN
from otx.algo.instance_segmentation.mmdet.models.roi_extractors import SingleRoIExtractor
from otx.algo.utils.mmengine_utils import InstanceData, load_checkpoint
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.config.data import TileConfig
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.instance_segmentation import InstanceSegBatchDataEntity, InstanceSegBatchPredEntity
from otx.core.data.entity.utils import stack_batch
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.metrics.mean_ap import MaskRLEMeanAPFMeasureCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.instance_segmentation import ExplainableOTXInstanceSegModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.label import LabelInfoTypes

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from torch.nn.modules import Module

    from otx.core.metrics import MetricCallable


class OTXMaskRCNN(ExplainableOTXInstanceSegModel):
    """MaskRCNN Model."""

    def __init__(
        self,
        label_info: LabelInfoTypes,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MaskRLEMeanAPFMeasureCallable,
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
        detector = self._build_model(num_classes=self.label_info.num_classes)
        detector.init_weights()
        self.classification_layers = self.get_classification_layers("model.")

        if self.load_from is not None:
            load_checkpoint(detector, self.load_from, map_location="cpu")
        return detector

    def _build_model(self, num_classes: int) -> OTXMaskRCNN:
        raise NotImplementedError

    def _customize_inputs(self, entity: InstanceSegBatchDataEntity) -> dict[str, Any]:
        if isinstance(entity.images, list):
            entity.images, entity.imgs_info = stack_batch(entity.images, entity.imgs_info, pad_size_divisor=32)
        inputs: dict[str, Any] = {}

        inputs["entity"] = entity
        inputs["mode"] = "loss" if self.training else "predict"

        return inputs

    def _customize_outputs(
        self,
        outputs: list[InstanceData] | dict,
        inputs: InstanceSegBatchDataEntity,
    ) -> InstanceSegBatchPredEntity | OTXBatchLossEntity:
        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = OTXBatchLossEntity()
            for loss_name, loss_value in outputs.items():
                if isinstance(loss_value, torch.Tensor):
                    losses[loss_name] = loss_value
                elif isinstance(loss_value, list):
                    losses[loss_name] = sum(_loss.mean() for _loss in loss_value)
            # pop acc from losses
            losses.pop("acc", None)
            return losses

        scores: list[torch.Tensor] = []
        bboxes: list[tv_tensors.BoundingBoxes] = []
        labels: list[torch.LongTensor] = []
        masks: list[tv_tensors.Mask] = []

        predictions = outputs["predictions"] if isinstance(outputs, dict) else outputs
        for img_info, prediction in zip(inputs.imgs_info, predictions):
            scores.append(prediction.scores)
            bboxes.append(
                tv_tensors.BoundingBoxes(
                    prediction.bboxes,
                    format="XYXY",
                    canvas_size=img_info.ori_shape,
                ),
            )
            output_masks = tv_tensors.Mask(
                prediction.masks,
                dtype=torch.bool,
            )
            masks.append(output_masks)
            labels.append(prediction.labels)

        if self.explain_mode:
            if not isinstance(outputs, dict):
                msg = f"Model output should be a dict, but got {type(outputs)}."
                raise ValueError(msg)

            if "feature_vector" not in outputs:
                msg = "No feature vector in the model output."
                raise ValueError(msg)

            if "saliency_map" not in outputs:
                msg = "No saliency maps in the model output."
                raise ValueError(msg)

            saliency_map = outputs["saliency_map"].detach().cpu().numpy()
            feature_vector = outputs["feature_vector"].detach().cpu().numpy()

            return InstanceSegBatchPredEntity(
                batch_size=len(predictions),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=scores,
                bboxes=bboxes,
                masks=masks,
                polygons=[],
                labels=labels,
                saliency_map=list(saliency_map),
                feature_vector=list(feature_vector),
            )

        return InstanceSegBatchPredEntity(
            batch_size=len(predictions),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=bboxes,
            masks=masks,
            polygons=[],
            labels=labels,
        )

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        if self.image_size is None:
            raise ValueError(self.image_size)

        input_size = self.tile_image_size if self.tile_config.enable_tiler else self.image_size

        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=input_size,
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
        meta_info_list = [meta_info] * len(inputs)
        return self.model.export(
            inputs,
            meta_info_list,
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_iseg_ckpt(state_dict, add_prefix)


class MaskRCNNResNet50(OTXMaskRCNN):
    """MaskRCNN with ResNet50 backbone."""

    load_from = (
        "https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_mstrain-poly_3x_coco/"
        "mask_rcnn_r50_fpn_mstrain-poly_3x_coco_20210524_201154-21b550bb.pth"
    )

    mean = (123.675, 116.28, 103.53)
    std = (58.395, 57.12, 57.375)

    def _build_model(self, num_classes: int) -> MaskRCNN:
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

        return MaskRCNN(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )


class MaskRCNNEfficientNet(OTXMaskRCNN):
    """MaskRCNN with efficientnet_b2b backbone."""

    load_from = (
        "https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/"
        "models/instance_segmentation/v2/efficientnet_b2b-mask_rcnn-576x576.pth"
    )

    mean = (123.675, 116.28, 103.53)
    std = (1.0, 1.0, 1.0)

    def _build_model(self, num_classes: int) -> MaskRCNN:
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

        test_cfg = DictConfig(
            {
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
            },
        )

        backbone = _build_model_including_pytorchcv(
            cfg={
                "type": "efficientnet_b2b",
                "out_indices": [2, 3, 4, 5],
                "frozen_stages": -1,
                "pretrained": True,
                "activation_cfg": {"type": "torch_swish"},
                "norm_cfg": {"type": "BN", "requires_grad": True},
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

        return MaskRCNN(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )


class MaskRCNNSwinT(OTXMaskRCNN):
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
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MaskRLEMeanAPFMeasureCallable,
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

    def _build_model(self, num_classes: int) -> MaskRCNN:
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

        return MaskRCNN(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
