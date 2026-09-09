# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MaskRCNN model implementations."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from torch import nn
from torchvision.ops import RoIAlign

from getitune.backend.lightning.exporter.base import ModelExporter
from getitune.backend.lightning.exporter.native import LightningModelExporter
from getitune.backend.lightning.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from getitune.backend.lightning.models.common.backbones import build_model_including_pytorchcv
from getitune.backend.lightning.models.common.losses import CrossEntropyLoss, CrossSigmoidFocalLoss, L1Loss
from getitune.backend.lightning.models.common.utils.assigners import MaxIoUAssigner
from getitune.backend.lightning.models.common.utils.coders import DeltaXYWHBBoxCoder
from getitune.backend.lightning.models.common.utils.prior_generators import AnchorGenerator
from getitune.backend.lightning.models.common.utils.samplers import RandomSampler
from getitune.backend.lightning.models.detection.necks import FPN
from getitune.backend.lightning.models.instance_segmentation.backbones.swin import SwinTransformer
from getitune.backend.lightning.models.instance_segmentation.base import LightningInstanceSegModel
from getitune.backend.lightning.models.instance_segmentation.heads import ConvFCBBoxHead, FCNMaskHead, RoIHead, RPNHead
from getitune.backend.lightning.models.instance_segmentation.losses import ROICriterion, RPNCriterion
from getitune.backend.lightning.models.instance_segmentation.segmentors.two_stage import TwoStageDetector
from getitune.backend.lightning.models.instance_segmentation.utils.roi_extractors import SingleRoIExtractor
from getitune.backend.lightning.models.modules.norm import build_norm_layer
from getitune.config.data import TileConfig
from getitune.metrics.fmeasure import MaskRLEMeanAPFMeasureCallable

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from getitune.backend.lightning.schedulers import LRSchedulerListCallable
    from getitune.metrics import MetricCallable
    from getitune.types import PathLike
    from getitune.types.label import LabelInfoTypes


class MaskRCNN(LightningInstanceSegModel):
    """Implementation of MaskRCNN for instance segmentation.

    Args:
        label_info (LabelInfoTypes): Information about the labels used in the model.
        data_input_params (DataInputParams | dict | None, optional): Parameters for the image data preprocessing.
            If None is given, default parameters for the specific model will be used.
        model_name (str, optional): Name of the model. Defaults to "maskrcnn_efficientnet_b2b".
        optimizer (OptimizerCallable, optional): Optimizer for the model. Defaults to DefaultOptimizerCallable.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): Scheduler for the model.
            Defaults to DefaultSchedulerCallable.
        metric (MetricCallable, optional): Metric for evaluating the model.
            Defaults to MaskRLEMeanAPFMeasureCallable.
        torch_compile (bool, optional): Whether to use torch compile. Defaults to False.
        tile_config (TileConfig, optional): Configuration for tiling. Defaults to TileConfig(enable_tiler=False).
        pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
        pretrained_weights (PathLike | None, optional): Path to the pretrained weights file. When None is passed,
            the default pretrained weights will be utilized for fine-tuning. Defaults to None.
    """

    _nms_always_embedded: ClassVar[bool] = True
    pretrained_urls: ClassVar[dict[str, str]] = {
        "maskrcnn_efficientnet_b2b": "https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/"
        "models/instance_segmentation/v2/efficientnet_b2b-mask_rcnn-576x576.pth",
        "maskrcnn_swin_tiny": "https://download.openmmlab.com/mmdetection/v2.0/swin/"
        "mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco/"
        "mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_20210908_165006-90a4008c.pth",
    }

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams | dict | None = None,
        model_name: Literal[
            "maskrcnn_efficientnet_b2b",
            "maskrcnn_swin_tiny",
        ] = "maskrcnn_efficientnet_b2b",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MaskRLEMeanAPFMeasureCallable,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
        pretrained: bool = True,
        pretrained_weights: PathLike | None = None,
        export_nms: bool = False,
    ) -> None:
        super().__init__(
            label_info=label_info,
            data_input_params=data_input_params,
            model_name=model_name,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            tile_config=tile_config,
            pretrained=pretrained,
            pretrained_weights=pretrained_weights,
            export_nms=export_nms,
        )

    def _create_model(self, num_classes: int | None = None) -> MaskRCNN:
        num_classes = num_classes if num_classes is not None else self.num_classes

        # TODO(Kirill): deprecate train_cfg/test_cfg
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

        rcnn_sampler = RandomSampler(
            add_gt_as_proposals=True,
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
        )

        backbone = self._build_backbone()
        neck = FPN(model_name=self.model_name)
        rpn_bbox_coder = DeltaXYWHBBoxCoder(
            target_means=(0.0, 0.0, 0.0, 0.0),
            target_stds=(1.0, 1.0, 1.0, 1.0),
        )

        rpn_head = RPNHead(
            model_name=self.model_name,
            anchor_generator=AnchorGenerator(
                strides=[4, 8, 16, 32, 64],
                ratios=[0.5, 1.0, 2.0],
                scales=[8],
            ),
            bbox_coder=rpn_bbox_coder,
            assigner=rpn_assigner,
            sampler=rpn_sampler,
            train_cfg=train_cfg["rpn"],
            test_cfg=test_cfg["rpn"],
        )

        roi_bbox_coder = DeltaXYWHBBoxCoder(
            target_means=(0.0, 0.0, 0.0, 0.0),
            target_stds=(0.1, 0.1, 0.2, 0.2),
        )

        bbox_head = ConvFCBBoxHead(
            model_name=self.model_name,
            num_classes=num_classes,
            bbox_coder=roi_bbox_coder,
        )

        bbox_roi_extractor = SingleRoIExtractor(
            featmap_strides=[4, 8, 16, 32],
            out_channels=rpn_head.feat_channels,
            roi_layer=RoIAlign(
                output_size=7,
                sampling_ratio=0,
                aligned=True,
                spatial_scale=1.0,
            ),
        )

        mask_roi_extractor = SingleRoIExtractor(
            featmap_strides=[4, 8, 16, 32],
            out_channels=rpn_head.feat_channels,
            roi_layer=RoIAlign(
                output_size=14,
                sampling_ratio=0,
                aligned=True,
                spatial_scale=1.0,
            ),
        )

        mask_head = FCNMaskHead(
            conv_out_channels=rpn_head.feat_channels,
            in_channels=rpn_head.feat_channels,
            num_classes=num_classes,
            num_convs=4,
        )

        roi_head = RoIHead(
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            assigner=rcnn_assigner,
            sampler=rcnn_sampler,
        )

        rpn_criterion = RPNCriterion(
            bbox_coder=DeltaXYWHBBoxCoder(
                target_means=(0.0, 0.0, 0.0, 0.0),
                target_stds=(1.0, 1.0, 1.0, 1.0),
            ),
            loss_bbox=L1Loss(loss_weight=1.0),
            loss_cls=CrossEntropyLoss(loss_weight=1.0, use_sigmoid=True),
        )

        roi_criterion = ROICriterion(
            num_classes=num_classes,
            bbox_coder=DeltaXYWHBBoxCoder(
                target_means=(0.0, 0.0, 0.0, 0.0),
                target_stds=(0.1, 0.1, 0.2, 0.2),
            ),
            loss_bbox=L1Loss(loss_weight=1.0),
            # TODO(someone): performance of CrossSigmoidFocalLoss is worse without mmcv
            # https://github.com/openvinotoolkit/training_extensions/pull/3431
            loss_cls=CrossSigmoidFocalLoss(loss_weight=1.0, use_sigmoid=False),
            loss_mask=CrossEntropyLoss(loss_weight=1.0, use_mask=True),
            class_agnostic=False,
        )

        return TwoStageDetector(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            roi_criterion=roi_criterion,
            rpn_criterion=rpn_criterion,
        )

    def _build_backbone(self) -> nn.Module:
        """Builds the backbone for the model."""
        backbone_cfg: dict[str, Any] = {
            "maskrcnn_swin_tiny": {
                "drop_path_rate": 0.2,
                "patch_norm": True,
                "convert_weights": True,
            },
            "maskrcnn_efficientnet_b2b": {
                "type": "efficientnet_b2b",
                "out_indices": [2, 3, 4, 5],
                "frozen_stages": -1,
                "activation": nn.SiLU,
                "normalization": partial(build_norm_layer, nn.BatchNorm2d, requires_grad=True),
            },
        }

        if "efficientnet" in self.model_name:
            cfg = backbone_cfg[self.model_name]
            return build_model_including_pytorchcv(cfg=cfg)

        if "swin" in self.model_name:
            return SwinTransformer(
                **backbone_cfg[self.model_name],
            )

        msg = ValueError(f"Model {self.model_name} is not supported.")
        raise msg

    @property
    def _exporter(self) -> ModelExporter:
        """Creates ModelExporter object that can export the model."""
        return LightningModelExporter(
            task_level_export_parameters=self._export_parameters,
            data_input_params=self.data_input_params,
            resize_mode="fit_to_window",
            pad_value=0,
            swap_rgb=False,
            via_onnx=True,
            onnx_export_configuration={
                "input_names": ["image"],
                "output_names": ["boxes", "labels", "masks"],
                "dynamic_axes": {
                    "image": {0: "batch"},
                    "boxes": {0: "batch", 1: "num_dets"},
                    "labels": {0: "batch", 1: "num_dets"},
                    "masks": {0: "batch", 1: "num_dets", 2: "height", 3: "width"},
                },
                "opset_version": 18,
                "autograd_inlining": False,
                "dynamo": False,
            },
            output_names=["boxes", "labels", "masks", "feature_vector", "saliency_map"] if self.explain_mode else None,
        )

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

        if self.model_name == "maskrcnn_swin_t":
            return {"model_type": "transformer"}

        return {}

    @property
    def _default_preprocessing_params(self) -> DataInputParams | dict[str, DataInputParams]:
        return {
            "maskrcnn_efficientnet_b2b": DataInputParams(
                input_size=(1024, 1024), mean=(0.485, 0.456, 0.406), std=(1.0, 1.0, 1.0)
            ),
            "maskrcnn_swin_tiny": DataInputParams(
                input_size=(1344, 1344), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        }
