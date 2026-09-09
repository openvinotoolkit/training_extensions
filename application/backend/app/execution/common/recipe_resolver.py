# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum
from pathlib import Path
from typing import ClassVar

from app.supported_models.timm import TimmManifestProvider


class ModelStatus(StrEnum):
    """Enum for model status."""

    SPEED = "speed"
    BALANCE = "balance"
    ACCURACY = "accuracy"
    DEPRECATED = "deprecated"
    ACTIVE = "active"


class RecipeResolver:
    """Resolves a Geti model manifest to a concrete getitune recipe file path."""

    TEMPLATE_ID_MAPPING: ClassVar[dict[str, dict]] = {
        # MULTI_CLASS_CLS
        "image-classification-vit-tiny": {
            "recipe_path": "classification/multi_class_cls/vit_tiny.yaml",
            "status": ModelStatus.BALANCE,
            "default": False,
        },
        "image-classification-dinov2": {
            "recipe_path": "classification/multi_class_cls/dino_v2.yaml",
            "status": ModelStatus.ACCURACY,
            "default": False,
        },
        "image-classification-efficientnet-b0": {
            "recipe_path": "classification/multi_class_cls/efficientnet_b0.yaml",
            "status": ModelStatus.ACTIVE,
            "default": True,
        },
        "image-classification-efficientnet-v2-s": {
            "recipe_path": "classification/multi_class_cls/efficientnet_v2.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "image-classification-mobilenet-v3-large": {
            "recipe_path": "classification/multi_class_cls/mobilenet_v3_large.yaml",
            "status": ModelStatus.SPEED,
            "default": False,
        },
        "image-classification-efficientnet-b3": {
            "recipe_path": "classification/multi_class_cls/efficientnet_b3.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "image-classification-yolo26-n": {
            "recipe_path": "classification/multi_class_cls/yolo26_n_cls.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "image-classification-yolo26-s": {
            "recipe_path": "classification/multi_class_cls/yolo26_s_cls.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "image-classification-yolo26-m": {
            "recipe_path": "classification/multi_class_cls/yolo26_m_cls.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "image-classification-yolo26-l": {
            "recipe_path": "classification/multi_class_cls/yolo26_l_cls.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "image-classification-yolo26-x": {
            "recipe_path": "classification/multi_class_cls/yolo26_x_cls.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        # DETECTION
        "object-detection-atss-mobilenet-v2": {
            "recipe_path": "detection/atss_mobilenetv2.yaml",
            "status": ModelStatus.ACTIVE,
            "default": True,
        },
        "object-detection-ssd-mobilenet-v2": {
            "recipe_path": "detection/ssd_mobilenetv2.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "object-detection-yolox-x": {
            "recipe_path": "detection/yolox_x.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "object-detection-yolox-l": {
            "recipe_path": "detection/yolox_l.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "object-detection-yolox-s": {
            "recipe_path": "detection/yolox_s.yaml",
            "status": ModelStatus.SPEED,
            "default": False,
        },
        "object-detection-yolox-tiny": {
            "recipe_path": "detection/yolox_tiny.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "object-detection-rt-detr-r50": {
            "recipe_path": "detection/rtdetr_50.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "object-detection-dfine-m": {
            "recipe_path": "detection/deim_dfine_m.yaml",
            "status": ModelStatus.BALANCE,
            "default": False,
        },
        "object-detection-dfine-l": {
            "recipe_path": "detection/deim_dfine_l.yaml",
            "status": ModelStatus.ACCURACY,
            "default": False,
        },
        "object-detection-dfine-x": {
            "recipe_path": "detection/deim_dfine_x.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "object-detection-rfdetr-n": {
            "recipe_path": "detection/rfdetr_nano.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "object-detection-rfdetr-s": {
            "recipe_path": "detection/rfdetr_small.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "object-detection-rfdetr-m": {
            "recipe_path": "detection/rfdetr_medium.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "object-detection-rfdetr-l": {
            "recipe_path": "detection/rfdetr_large.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "object-detection-edgecrafter-s": {
            "recipe_path": "detection/edgecrafter_s.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "object-detection-edgecrafter-m": {
            "recipe_path": "detection/edgecrafter_m.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "object-detection-edgecrafter-l": {
            "recipe_path": "detection/edgecrafter_l.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "object-detection-edgecrafter-x": {
            "recipe_path": "detection/edgecrafter_x.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "object-detection-dinov3-detr-s": {
            "recipe_path": "detection/deimv2_s.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "object-detection-dinov3-detr-m": {
            "recipe_path": "detection/deimv2_m.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "object-detection-dinov3-detr-l": {
            "recipe_path": "detection/deimv2_l.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "object-detection-yolo26-n": {
            "recipe_path": "detection/yolo26_n.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "object-detection-yolo26-s": {
            "recipe_path": "detection/yolo26_s.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "object-detection-yolo26-m": {
            "recipe_path": "detection/yolo26_m.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "object-detection-yolo26-l": {
            "recipe_path": "detection/yolo26_l.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "object-detection-yolo26-x": {
            "recipe_path": "detection/yolo26_x.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "object-detection-yolo11-n": {
            "recipe_path": "detection/yolo11_n.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "object-detection-yolo11-s": {
            "recipe_path": "detection/yolo11_s.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "object-detection-yolo11-m": {
            "recipe_path": "detection/yolo11_m.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "object-detection-yolo11-l": {
            "recipe_path": "detection/yolo11_l.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "object-detection-yolo11-x": {
            "recipe_path": "detection/yolo11_x.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "object-detection-yolo12-n": {
            "recipe_path": "detection/yolo12_n.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "object-detection-yolo12-s": {
            "recipe_path": "detection/yolo12_s.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "object-detection-yolo12-m": {
            "recipe_path": "detection/yolo12_m.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "object-detection-yolo12-l": {
            "recipe_path": "detection/yolo12_l.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "object-detection-yolo12-x": {
            "recipe_path": "detection/yolo12_x.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        # INSTANCE_SEGMENTATION
        "instance-segmentation-mask-rcnn-swin-t": {
            "recipe_path": "instance_segmentation/maskrcnn_swint.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "instance-segmentation-mask-rcnn-efficientnet-b2": {
            "recipe_path": "instance_segmentation/maskrcnn_efficientnetb2b.yaml",
            "status": ModelStatus.ACTIVE,
            "default": True,
        },
        "instance-segmentation-rtmdet-tiny": {
            "recipe_path": "instance_segmentation/rtmdet_inst_tiny.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "instance-segmentation-mask-rcnn-resnet50": {
            "recipe_path": "instance_segmentation/maskrcnn_r50.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "instance-segmentation-rfdetr-n": {
            "recipe_path": "instance_segmentation/rfdetr_seg_nano.yaml",
            "status": ModelStatus.SPEED,
            "default": False,
        },
        "instance-segmentation-rfdetr-s": {
            "recipe_path": "instance_segmentation/rfdetr_seg_small.yaml",
            "status": ModelStatus.SPEED,
            "default": False,
        },
        "instance-segmentation-rfdetr-m": {
            "recipe_path": "instance_segmentation/rfdetr_seg_medium.yaml",
            "status": ModelStatus.BALANCE,
            "default": False,
        },
        "instance-segmentation-rfdetr-l": {
            "recipe_path": "instance_segmentation/rfdetr_seg_large.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "instance-segmentation-rfdetr-xl": {
            "recipe_path": "instance_segmentation/rfdetr_seg_xlarge.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "instance-segmentation-rfdetr-2xl": {
            "recipe_path": "instance_segmentation/rfdetr_seg_2xlarge.yaml",
            "status": ModelStatus.ACCURACY,
            "default": False,
        },
        "instance-segmentation-yolo26-n": {
            "recipe_path": "instance_segmentation/yolo26_n_seg.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "instance-segmentation-yolo26-s": {
            "recipe_path": "instance_segmentation/yolo26_s_seg.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "instance-segmentation-yolo26-m": {
            "recipe_path": "instance_segmentation/yolo26_m_seg.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "instance-segmentation-yolo26-l": {
            "recipe_path": "instance_segmentation/yolo26_l_seg.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "instance-segmentation-yolo26-x": {
            "recipe_path": "instance_segmentation/yolo26_x_seg.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "instance-segmentation-yolo11-n": {
            "recipe_path": "instance_segmentation/yolo11_n_seg.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "instance-segmentation-yolo11-s": {
            "recipe_path": "instance_segmentation/yolo11_s_seg.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "instance-segmentation-yolo11-m": {
            "recipe_path": "instance_segmentation/yolo11_m_seg.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "instance-segmentation-yolo11-l": {
            "recipe_path": "instance_segmentation/yolo11_l_seg.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
        "instance-segmentation-yolo11-x": {
            "recipe_path": "instance_segmentation/yolo11_x_seg.yaml",
            "status": ModelStatus.ACTIVE,
            "default": False,
        },
    }

    def __init__(self, recipe_root: Path) -> None:
        self._recipe_root = recipe_root

    def resolve(
        self,
        model_manifest_id: str,
        sub_task_type: str | None,
    ) -> Path:
        """Resolve the recipe file path for a model manifest.

        Args:
            model_manifest_id: Geti model manifest identifier.
            sub_task_type: Sub-task discriminator (e.g. "MULTI_LABEL_CLS").

        Returns:
            Resolved, existing recipe path.

        Raises:
            KeyError: If model_manifest_id is not registered.
            FileNotFoundError: If the resolved recipe file does not exist.
        """
        if TimmManifestProvider.is_timm_id(model_manifest_id) and sub_task_type:
            return self._recipe_root / "classification" / sub_task_type.lower() / "timm_generic.yaml"

        try:
            entry = self.TEMPLATE_ID_MAPPING[model_manifest_id]
        except KeyError as exc:
            msg = f"Unknown model manifest id: '{model_manifest_id}'"
            raise KeyError(msg) from exc

        path = self._recipe_root / str(entry["recipe_path"])
        path = self._apply_sub_task_variant(path, sub_task_type)

        if not path.exists():
            msg = f"Recipe file not found: {path}"
            raise FileNotFoundError(msg)
        return path

    def _apply_sub_task_variant(self, path: Path, sub_task_type: str | None) -> Path:
        """
        Select the classification sub-task recipe variant, for either backend.

        Classification manifests map to a single architecture recipe (e.g.
        ``yolo26_m_cls.yaml``) shared by both multi-class and multi-label
        tasks. ``sub_task_type`` (e.g. ``"MULTI_LABEL_CLS"``) picks the
        matching recipe subdirectory.
        """
        if sub_task_type and "_cls" in path.parent.name:
            return self._recipe_root / "classification" / sub_task_type.lower() / path.name
        return path
