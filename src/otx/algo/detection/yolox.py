# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""YOLOX model implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from mmengine.structures import InstanceData
from omegaconf import DictConfig
from torchvision import tv_tensors

from otx.algo.detection.backbones.csp_darknet import CSPDarknet
from otx.algo.detection.heads.yolox_head import YOLOXHead
from otx.algo.detection.necks.yolox_pafpn import YOLOXPAFPN
from otx.algo.detection.ssd import SingleStageDetector
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.detection import DetBatchDataEntity, DetBatchPredEntity
from otx.core.data.entity.utils import stack_batch
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.model.detection import ExplainableOTXDetModel

if TYPE_CHECKING:
    from torch import Tensor, nn


class YOLOX(ExplainableOTXDetModel):
    """OTX Detection model class for YOLOX."""

    def _create_model(self) -> nn.Module:
        from mmengine.runner import load_checkpoint

        detector = self._build_model(num_classes=self.label_info.num_classes)
        detector.init_weights()
        self.classification_layers = self.get_classification_layers(prefix="model.")
        if self.load_from is not None:
            load_checkpoint(detector, self.load_from, map_location="cpu")
        return detector

    def _build_model(self, num_classes: int) -> nn.Module:
        raise NotImplementedError

    def _customize_inputs(self, entity: DetBatchDataEntity) -> dict[str, Any]:
        if isinstance(entity.images, list):
            entity.images = stack_batch(entity.images, pad_size_divisor=32, pad_value=114)
        inputs: dict[str, Any] = {}

        inputs["entity"] = entity
        inputs["mode"] = "loss" if self.training else "predict"

        return inputs

    def _customize_outputs(
        self,
        outputs: list[InstanceData] | dict,
        inputs: DetBatchDataEntity,
    ) -> DetBatchPredEntity | OTXBatchLossEntity:
        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = OTXBatchLossEntity()
            for k, v in outputs.items():
                if isinstance(v, list):
                    losses[k] = sum(v)
                elif isinstance(v, torch.Tensor):
                    losses[k] = v
                else:
                    msg = "Loss output should be list or torch.tensor but got {type(v)}"
                    raise TypeError(msg)
            return losses

        scores = []
        bboxes = []
        labels = []
        predictions = outputs["predictions"] if isinstance(outputs, dict) else outputs
        for img_info, prediction in zip(inputs.imgs_info, predictions):
            if not isinstance(prediction, InstanceData):
                raise TypeError(prediction)
            scores.append(prediction.scores)
            bboxes.append(
                tv_tensors.BoundingBoxes(
                    prediction.bboxes,
                    format="XYXY",
                    canvas_size=img_info.ori_shape,
                ),
            )
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

            return DetBatchPredEntity(
                batch_size=len(outputs),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=scores,
                bboxes=bboxes,
                labels=labels,
                saliency_map=saliency_map,
                feature_vector=feature_vector,
            )

        return DetBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=bboxes,
            labels=labels,
        )

    def get_classification_layers(
        self,
        prefix: str = "",
    ) -> dict[str, dict[str, int]]:
        """Return classification layer names by comparing two different number of classes models.

        TODO (sungchul): it can be merged to otx.core.utils.build.get_classification_layers.

        Args:
            config (DictConfig): Config for building model.
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

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        if self.image_size is None:
            raise ValueError(self.image_size)

        swap_rgb = not isinstance(self, YOLOXTINY)

        return OTXNativeModelExporter(
            via_onnx=True,
            onnx_export_configuration={
                "input_names": ["image"],
                "output_names": ["boxes", "labels"],
                "export_params": True,
                "opset_version": 11,
                "dynamic_axes": {
                    "image": {0: "batch", 2: "height", 3: "width"},
                    "boxes": {0: "batch", 1: "num_dets"},
                    "labels": {0: "batch", 1: "num_dets"},
                },
                "keep_initializers_as_inputs": False,
                "verbose": False,
                "autograd_inlining": False,
            },
            task_level_export_parameters=self._export_parameters,
            input_size=self.image_size,
            mean=self.mean,
            std=self.std,
            resize_mode="fit_to_window_letterbox",
            pad_value=114,
            swap_rgb=swap_rgb,
            output_names=["bboxes", "labels", "feature_vector", "saliency_map"] if self.explain_mode else None,
        )

    def forward_for_tracing(self, inputs: Tensor) -> list[InstanceData]:
        """Forward function for export."""
        shape = (int(inputs.shape[2]), int(inputs.shape[3]))
        meta_info = {
            "pad_shape": shape,
            "batch_input_shape": shape,
            "img_shape": shape,
            "scale_factor": (1.0, 1.0),
        }

        meta_info_list = [meta_info] * len(inputs)
        return self.model.export(inputs, meta_info_list, explain_mode=self.explain_mode)

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_det_ckpt(state_dict, add_prefix)


class YOLOXTINY(YOLOX):
    """YOLOX-TINY detector."""

    load_from = (
        "https://storage.openvinotoolkit.org/repositories/"
        "openvino_training_extensions/models/object_detection/v2/yolox_tiny_8x8.pth"
    )
    image_size = (1, 3, 416, 416)
    tile_image_size = (1, 3, 416, 416)
    mean = (123.675, 116.28, 103.53)
    std = (58.395, 57.12, 57.375)

    def _build_model(self, num_classes: int) -> SingleStageDetector:
        train_cfg: dict[str, Any] = {}
        test_cfg = DictConfig(
            {
                "nms": {"type": "nms", "iou_threshold": 0.65},
                "score_thr": 0.01,
                "max_per_img": 100,
            },
        )
        backbone = CSPDarknet(
            deepen_factor=0.33,
            widen_factor=0.375,
            out_indices=[2, 3, 4],
        )
        neck = YOLOXPAFPN(
            in_channels=[96, 192, 384],
            out_channels=96,
            num_csp_blocks=1,
        )
        bbox_head = YOLOXHead(
            num_classes=num_classes,
            in_channels=96,
            feat_channels=96,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        return SingleStageDetector(backbone, bbox_head, neck=neck, train_cfg=train_cfg, test_cfg=test_cfg)


class YOLOXS(YOLOX):
    """YOLOX-S detector."""

    load_from = (
        "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/"
        "yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth"
    )
    image_size = (1, 3, 640, 640)
    tile_image_size = (1, 3, 640, 640)
    mean = (0.0, 0.0, 0.0)
    std = (1.0, 1.0, 1.0)

    def _build_model(self, num_classes: int) -> SingleStageDetector:
        train_cfg = DictConfig({})
        test_cfg = DictConfig(
            {
                "nms": {"type": "nms", "iou_threshold": 0.65},
                "score_thr": 0.01,
                "max_per_img": 100,
            },
        )
        backbone = CSPDarknet(
            deepen_factor=0.33,
            widen_factor=0.5,
            out_indices=[2, 3, 4],
        )
        neck = YOLOXPAFPN(
            in_channels=[128, 256, 512],
            out_channels=128,
            num_csp_blocks=1,
        )
        bbox_head = YOLOXHead(
            num_classes=num_classes,
            in_channels=128,
            feat_channels=128,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        return SingleStageDetector(backbone, bbox_head, neck=neck, train_cfg=train_cfg, test_cfg=test_cfg)


class YOLOXL(YOLOX):
    """YOLOX-L detector."""

    load_from = (
        "https://download.openmmlab.com/mmdetection/v2.0/yolox/"
        "yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth"
    )
    image_size = (1, 3, 640, 640)
    tile_image_size = (1, 3, 640, 640)
    mean = (0.0, 0.0, 0.0)
    std = (1.0, 1.0, 1.0)

    def _build_model(self, num_classes: int) -> SingleStageDetector:
        train_cfg = DictConfig({})
        test_cfg = DictConfig(
            {
                "nms": {"type": "nms", "iou_threshold": 0.65},
                "score_thr": 0.01,
                "max_per_img": 100,
            },
        )
        backbone = CSPDarknet(
            deepen_factor=1.0,
            widen_factor=1.0,
            out_indices=[2, 3, 4],
        )
        neck = YOLOXPAFPN(
            in_channels=[256, 512, 1024],
            out_channels=256,
            num_csp_blocks=3,
        )
        bbox_head = YOLOXHead(
            num_classes=num_classes,
            in_channels=256,
            feat_channels=256,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        return SingleStageDetector(backbone, bbox_head, neck=neck, train_cfg=train_cfg, test_cfg=test_cfg)


class YOLOXX(YOLOX):
    """YOLOX-X detector."""

    load_from = (
        "https://download.openmmlab.com/mmdetection/v2.0/yolox/"
        "yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth"
    )
    image_size = (1, 3, 640, 640)
    tile_image_size = (1, 3, 640, 640)
    mean = (0.0, 0.0, 0.0)
    std = (1.0, 1.0, 1.0)

    def _build_model(self, num_classes: int) -> SingleStageDetector:
        train_cfg = DictConfig({})
        test_cfg = DictConfig(
            {
                "nms": {"type": "nms", "iou_threshold": 0.65},
                "score_thr": 0.01,
                "max_per_img": 100,
            },
        )
        backbone = CSPDarknet(
            deepen_factor=1.33,
            widen_factor=1.25,
            out_indices=[2, 3, 4],
        )
        neck = YOLOXPAFPN(
            in_channels=[320, 640, 1280],
            out_channels=320,
            num_csp_blocks=4,
        )
        bbox_head = YOLOXHead(
            num_classes=num_classes,
            in_channels=320,
            feat_channels=320,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        return SingleStageDetector(backbone, bbox_head, neck=neck, train_cfg=train_cfg, test_cfg=test_cfg)
