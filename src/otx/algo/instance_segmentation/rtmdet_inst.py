# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""RTMDetInst model implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mmengine.structures import InstanceData
from omegaconf import DictConfig

from otx.algo.detection.backbones.cspnext import CSPNeXt
from otx.algo.detection.heads.base_sampler import PseudoSampler
from otx.algo.detection.heads.distance_point_bbox_coder import DistancePointBBoxCoder
from otx.algo.detection.heads.dynamic_soft_label_assigner import DynamicSoftLabelAssigner
from otx.algo.detection.heads.point_generator import MlvlPointGenerator
from otx.algo.detection.losses.cross_entropy_loss import CrossEntropyLoss
from otx.algo.detection.losses.dice_loss import DiceLoss
from otx.algo.detection.losses.gfocal_loss import QualityFocalLoss
from otx.algo.detection.losses.iou_loss import GIoULoss
from otx.algo.detection.necks.cspnext_pafpn import CSPNeXtPAFPN
from otx.algo.instance_segmentation.mmdet.models.dense_heads.rtmdet_ins_head import RTMDetInsSepBNHead
from otx.algo.instance_segmentation.mmdet.models.detectors.rtmdet import RTMDet
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

if TYPE_CHECKING:
    import torch
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from torch.nn.modules import Module

    from otx.core.metrics import MetricCallable


class MMDetRTMDetInstTiny(MMDetInstanceSegCompatibleModel):
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
        self.image_size = (1, 3, 640, 640)
        self.tile_image_size = self.image_size

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

    def _build_model(self, num_classes: int) -> RTMDet:
        train_cfg = {
            "assigner": DynamicSoftLabelAssigner(topk=13),
            "sampler": PseudoSampler(),
            "allowed_border": -1,
            "pos_weight": -1,
            "debug": False,
        }

        test_cfg = DictConfig(
            {
                "nms": {"type": "nms", "iou_threshold": 0.5},
                "score_thr": 0.05,
                "mask_thr_binary": 0.5,
                "max_per_img": 100,
                "min_bbox_size": 0,
                "nms_pre": 300,
            },
        )

        data_preprocessor = DetDataPreprocessor(
            mean=self.mean,
            std=self.std,
            pad_value=114,
            bgr_to_rgb=False,
            pad_mask=True,
            pad_size_divisor=32,
            non_blocking=True,
        )

        backbone = CSPNeXt(
            arch="P5",
            expand_ratio=0.5,
            deepen_factor=0.167,
            widen_factor=0.375,
            channel_attention=True,
            norm_cfg={"type": "BN"},
            act_cfg={"type": "SiLU", "inplace": True},
        )

        neck = CSPNeXtPAFPN(
            in_channels=(96, 192, 384),
            out_channels=96,
            num_csp_blocks=1,
            expand_ratio=0.5,
            norm_cfg={"type": "BN"},
            act_cfg={"type": "SiLU", "inplace": True},
        )

        bbox_head = RTMDetInsSepBNHead(
            num_classes=num_classes,
            in_channels=96,
            stacked_convs=2,
            share_conv=True,
            pred_kernel_size=1,
            feat_channels=96,
            act_cfg={"type": "SiLU", "inplace": True},
            norm_cfg={"type": "BN", "requires_grad": True},
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

        return RTMDet(
            data_preprocessor=data_preprocessor,
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )

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
