# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""YOLOX model implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig

from otx.algo.detection.backbones.csp_darknet import CSPDarknet
from otx.algo.detection.heads.sim_ota_assigner import SimOTAAssigner
from otx.algo.detection.heads.yolox_head import YOLOXHead
from otx.algo.detection.necks.yolox_pafpn import YOLOXPAFPN
from otx.algo.detection.ssd import SingleStageDetector
from otx.algo.utils.mmengine_utils import InstanceData
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.data.entity.detection import DetBatchDataEntity
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.model.detection import ExplainableOTXDetModel
from otx.core.types.export import OTXExportFormatType
from otx.core.types.precision import OTXPrecisionType

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor


class YOLOX(ExplainableOTXDetModel):
    """OTX Detection model class for YOLOX."""

    def _build_model(self, num_classes: int) -> SingleStageDetector:
        raise NotImplementedError

    def _customize_inputs(
        self,
        entity: DetBatchDataEntity,
        pad_size_divisor: int = 32,
        pad_value: int = 114,  # YOLOX uses 114 as pad_value
    ) -> dict[str, Any]:
        return super()._customize_inputs(entity=entity, pad_size_divisor=pad_size_divisor, pad_value=pad_value)

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

    def export(
        self,
        output_dir: Path,
        base_name: str,
        export_format: OTXExportFormatType,
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
        to_exportable_code: bool = False,
    ) -> Path:
        """Export this model to the specified output directory.

        This is required to patch otx.algo.detection.backbones.csp_darknet.Focus.forward to export forward.

        Args:
            output_dir (Path): directory for saving the exported model
            base_name: (str): base name for the exported model file. Extension is defined by the target export format
            export_format (OTXExportFormatType): format of the output model
            precision (OTXExportPrecisionType): precision of the output model

        Returns:
            Path: path to the exported model.
        """
        # patch otx.algo.detection.backbones.csp_darknet.Focus.forward
        orig_focus_forward = self.model.backbone.stem.forward
        try:
            self.model.backbone.stem.forward = self.model.backbone.stem.export
            return super().export(output_dir, base_name, export_format, precision, to_exportable_code)
        finally:
            self.model.backbone.stem.forward = orig_focus_forward

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

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.") -> dict:
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
        train_cfg: dict[str, Any] = {"assigner": SimOTAAssigner(center_radius=2.5)}
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
        train_cfg: dict[str, Any] = {"assigner": SimOTAAssigner(center_radius=2.5)}
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
        train_cfg: dict[str, Any] = {"assigner": SimOTAAssigner(center_radius=2.5)}
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
        train_cfg: dict[str, Any] = {"assigner": SimOTAAssigner(center_radius=2.5)}
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
