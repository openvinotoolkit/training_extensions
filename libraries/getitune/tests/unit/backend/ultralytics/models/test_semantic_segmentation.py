# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Ultralytics semantic segmentation model wrapper.

Covers ``getitune.backend.ultralytics.models.semantic_segmentation``.
"""

from __future__ import annotations

from getitune.backend.ultralytics.models import UltralyticsSemanticSegModel
from getitune.types.label import SegLabelInfo


class TestSemanticSegmentationModel:
    """Tests for the semantic segmentation model wrapper."""

    def test_task_is_semantic(self) -> None:
        model = UltralyticsSemanticSegModel(model_name="yolo26n-sem.yaml", pretrained=False, label_info=3)
        assert model.task == "semantic"

    def test_default_preprocessing_is_512_identity(self) -> None:
        model = UltralyticsSemanticSegModel(model_name="yolo26n-sem", pretrained=False, label_info=3)
        params = model.data_input_params
        assert params.input_size == (512, 512)
        assert params.mean == (0.0, 0.0, 0.0)
        assert params.std == (1.0, 1.0, 1.0)

    def test_label_info_dispatch_returns_seg_label_info(self) -> None:
        model = UltralyticsSemanticSegModel(
            model_name="yolo26n-sem.yaml",
            pretrained=False,
            label_info=3,
        )
        assert isinstance(model.label_info, SegLabelInfo)
        assert model.label_info.num_classes == 3

    def test_export_parameters(self) -> None:
        model = UltralyticsSemanticSegModel(
            model_name="yolo26n-sem.yaml",
            pretrained=False,
            label_info=3,
        )
        params = model._export_parameters
        assert params.model_type == "Segmentation"
        assert params.task_type == "segmentation"
        assert params.nms_execute is False
        assert isinstance(params.label_info, SegLabelInfo)
