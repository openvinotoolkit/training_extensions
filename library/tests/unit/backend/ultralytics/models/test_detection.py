# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Ultralytics detection model wrapper.

Covers ``getitune.backend.ultralytics.models.detection``.
"""

from __future__ import annotations

import pytest

from getitune.backend.ultralytics.models import UltralyticsDetectionModel
from getitune.backend.ultralytics.trainers.detection import DetectionTrainer
from getitune.backend.ultralytics.validators.detection import DetectionValidator
from getitune.types.export import TaskLevelExportParameters
from getitune.types.label import LabelInfo


def _label_info() -> LabelInfo:
    return LabelInfo(label_names=["cat", "dog"], label_ids=["0", "1"], label_groups=[["cat", "dog"]])


class TestDetectionModel:
    """Tests for the detection model wrapper."""

    def test_task_is_detect(self) -> None:
        model = UltralyticsDetectionModel(model_name="yolo26n.yaml", pretrained=False, label_info=_label_info())
        assert model.task == "detect"

    def test_trainer_and_validator_classes(self) -> None:
        assert UltralyticsDetectionModel.trainer_cls is DetectionTrainer
        assert UltralyticsDetectionModel.validator_cls is DetectionValidator

    def test_default_preprocessing_is_640_identity(self) -> None:
        model = UltralyticsDetectionModel(model_name="yolo26n", pretrained=False, label_info=_label_info())
        params = model.data_input_params
        assert params.input_size == (640, 640)
        assert params.mean == (0.0, 0.0, 0.0)
        assert params.std == (1.0, 1.0, 1.0)

    def test_export_parameters(self) -> None:
        model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())
        params = model._export_parameters
        assert isinstance(params, TaskLevelExportParameters)
        assert params.model_type == "YOLO11"
        assert params.task_type == "detection"
        assert params.label_info == _label_info()
        assert params.iou_threshold == 0.5

    @pytest.mark.parametrize(("export_nms", "expected_nms_execute"), [(False, True), (True, False)])
    def test_export_nms_controls_deferred_nms_metadata(
        self, export_nms: bool, expected_nms_execute: bool | None
    ) -> None:
        model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info(), export_nms=export_nms)

        assert model.export_nms is export_nms
        assert model._export_parameters.nms_execute is expected_nms_execute

    @pytest.mark.parametrize(
        "variant",
        [
            "yolo26n",
            "yolo26s",
            "yolo26m",
            "yolo26l",
            "yolo26x",
            "yolo11n",
            "yolo11s",
            "yolo11m",
            "yolo11l",
            "yolo11x",
            "yolo12n",
            "yolo12s",
            "yolo12m",
            "yolo12l",
            "yolo12x",
        ],
    )
    def test_pretrained_weights_cover_all_supported_variants(self, variant: str) -> None:
        weights = UltralyticsDetectionModel._pretrained_weights
        assert variant in weights
        assert f"{variant}.pt" in weights[variant]

    def test_preprocessing_params_cover_all_variants(self) -> None:
        model = UltralyticsDetectionModel(model_name="yolo26n.yaml", pretrained=False, label_info=_label_info())
        defaults = model._default_preprocessing_params
        for variant in UltralyticsDetectionModel._pretrained_weights:
            assert variant in defaults, f"No preprocessing params for detection variant {variant!r}"
