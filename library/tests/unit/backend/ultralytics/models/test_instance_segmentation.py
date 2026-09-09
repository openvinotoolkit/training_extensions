# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Ultralytics instance-segmentation model wrapper.

Covers ``getitune.backend.ultralytics.models.instance_segmentation``.
"""

from __future__ import annotations

import pytest

from getitune.backend.ultralytics.models import UltralyticsInstSegModel
from getitune.backend.ultralytics.trainers.instance_segmentation import SegmentationTrainer
from getitune.backend.ultralytics.validators.instance_segmentation import SegmentationValidator
from getitune.types.export import TaskLevelExportParameters
from getitune.types.label import LabelInfo


def _label_info() -> LabelInfo:
    return LabelInfo(label_names=["cat", "dog"], label_ids=["0", "1"], label_groups=[["cat", "dog"]])


class TestInstanceSegmentationModel:
    """Tests for the instance-segmentation model wrapper."""

    def test_task_is_segment(self) -> None:
        model = UltralyticsInstSegModel(model_name="yolo26n-seg.yaml", pretrained=False, label_info=_label_info())
        assert model.task == "segment"

    def test_trainer_and_validator_classes(self) -> None:
        assert UltralyticsInstSegModel.trainer_cls is SegmentationTrainer
        assert UltralyticsInstSegModel.validator_cls is SegmentationValidator

    def test_default_preprocessing_is_640_identity(self) -> None:
        model = UltralyticsInstSegModel(model_name="yolo26n-seg", pretrained=False, label_info=_label_info())
        params = model.data_input_params
        assert params.input_size == (640, 640)
        assert params.mean == (0.0, 0.0, 0.0)
        assert params.std == (1.0, 1.0, 1.0)

    def test_export_parameters_default_thresholds(self) -> None:
        """With no explicit overrides, confidence/iou fall back to 0.25/0.5."""
        model = UltralyticsInstSegModel(model_name="yolo26n-seg", label_info=_label_info())
        params = model._export_parameters
        assert isinstance(params, TaskLevelExportParameters)
        assert params.model_type == "YOLO-seg"
        assert params.task_type == "instance_segmentation"
        assert params.label_info == _label_info()
        assert params.confidence_threshold == 0.25
        assert params.iou_threshold == 0.5
        assert params.nms_execute is True

    def test_export_parameters_omit_nms_metadata_when_embedded(self) -> None:
        model = UltralyticsInstSegModel(model_name="yolo26n-seg", label_info=_label_info(), export_nms=True)

        assert model._export_parameters.nms_execute is False

    def test_export_parameters_use_export_args_override(self) -> None:
        model = UltralyticsInstSegModel(model_name="yolo26n-seg", label_info=_label_info())
        model._export_args = {"confidence_threshold": 0.4, "iou_threshold": 0.6}
        params = model._export_parameters
        assert params.confidence_threshold == 0.4
        assert params.iou_threshold == 0.6

    def test_export_parameters_use_extra_overrides_fallback(self) -> None:
        model = UltralyticsInstSegModel(
            model_name="yolo26n-seg",
            label_info=_label_info(),
            extra_overrides={"conf": 0.35, "iou": 0.55},
        )
        params = model._export_parameters
        assert params.confidence_threshold == 0.35
        assert params.iou_threshold == 0.55

    @pytest.mark.parametrize(
        "variant",
        [
            "yolo26n-seg",
            "yolo26s-seg",
            "yolo26m-seg",
            "yolo26l-seg",
            "yolo26x-seg",
            "yolo11n-seg",
            "yolo11s-seg",
            "yolo11m-seg",
            "yolo11l-seg",
            "yolo11x-seg",
        ],
    )
    def test_pretrained_weights_cover_all_supported_variants(self, variant: str) -> None:
        weights = UltralyticsInstSegModel._pretrained_weights
        assert variant in weights
        assert f"{variant}.pt" in weights[variant]

    def test_preprocessing_params_cover_all_variants(self) -> None:
        model = UltralyticsInstSegModel(model_name="yolo26n-seg.yaml", pretrained=False, label_info=_label_info())
        defaults = model._default_preprocessing_params
        for variant in UltralyticsInstSegModel._pretrained_weights:
            assert variant in defaults, f"No preprocessing params for seg variant {variant!r}"
