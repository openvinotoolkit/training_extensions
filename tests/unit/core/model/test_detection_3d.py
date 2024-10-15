# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for keypoint detection model entity."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from otx.algo.object_detection_3d.monodetr3d import MonoDETR3D
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.object_detection_3d import Det3DBatchDataEntity, Det3DBatchPredEntity
from otx.core.metrics.average_precision_3d import KittiMetric
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.types.label import LabelInfo

if TYPE_CHECKING:
    from otx.core.model.detection_3d import OTX3DDetectionModel


class TestOTX3DDetectionModel:
    @pytest.fixture()
    def model(self, label_info, optimizer, scheduler, metric, torch_compile) -> OTX3DDetectionModel:
        return MonoDETR3D(label_info, "monodetr_50", (1280, 384), optimizer, scheduler, metric, torch_compile)

    @pytest.fixture()
    def batch_data_entity(self, model) -> Det3DBatchDataEntity:
        return model.get_dummy_input(2)

    @pytest.fixture()
    def label_info(self) -> LabelInfo:
        return LabelInfo(
            label_names=["label_0", "label_1"],
            label_groups=[["label_0", "label_1"]],
        )

    @pytest.fixture()
    def optimizer(self):
        return DefaultOptimizerCallable

    @pytest.fixture()
    def scheduler(self):
        return DefaultSchedulerCallable

    @pytest.fixture()
    def metric(self):
        return KittiMetric

    @pytest.fixture()
    def torch_compile(self):
        return False

    def test_export_parameters(self, model):
        params = model._export_parameters
        assert params.model_type == "mono_3d_det"
        assert params.task_type == "3d_detection"

    @pytest.mark.parametrize(
        ("label_info", "expected_label_info"),
        [
            (
                LabelInfo(label_names=["label1", "label2", "label3"], label_groups=[["label1", "label2", "label3"]]),
                LabelInfo(label_names=["label1", "label2", "label3"], label_groups=[["label1", "label2", "label3"]]),
            ),
            (LabelInfo.from_num_classes(num_classes=5), LabelInfo.from_num_classes(num_classes=5)),
        ],
    )
    def test_dispatch_label_info(self, model, label_info, expected_label_info):
        result = model._dispatch_label_info(label_info)
        assert result == expected_label_info

    def test_init(self, model):
        assert model.num_classes == 2

    def test_customize_inputs(self, model, batch_data_entity):
        customized_inputs = model._customize_inputs(batch_data_entity)
        assert customized_inputs["images"].shape == (2, 3, model.input_size[0], model.input_size[1])
        assert "mode" in customized_inputs
        assert "calibs" in customized_inputs
        assert customized_inputs["calibs"].shape == (2, 3, 4)

    def test_customize_outputs_training(self, model, batch_data_entity):
        outputs = {"loss": torch.tensor(0.5)}
        customized_outputs = model._customize_outputs(outputs, batch_data_entity)
        assert isinstance(customized_outputs, OTXBatchLossEntity)
        assert customized_outputs["loss"] == torch.tensor(0.5)

    def test_customize_outputs_predict(self, model, batch_data_entity):
        model.training = False
        outputs = {
            "scores": torch.randn(2, 50, 2),
            "boxes_3d": torch.randn(2, 50, 6),
            "boxes": torch.randn(2, 50, 4),
            "size_3d": torch.randn(2, 50, 3),
            "depth": torch.randn(2, 50, 2),
            "heading_angle": torch.randn(2, 50, 24),
        }
        customized_outputs = model._customize_outputs(outputs, batch_data_entity)
        assert isinstance(customized_outputs, Det3DBatchPredEntity)
        assert hasattr(customized_outputs, "scores")
        assert hasattr(customized_outputs, "heading_angle")
        assert hasattr(customized_outputs, "boxes")
        assert hasattr(customized_outputs, "size_2d")
        assert len(customized_outputs.boxes_3d) == len(customized_outputs.scores)

    def test_dummy_input(self, model: OTX3DDetectionModel):
        batch_size = 2
        batch = model.get_dummy_input(batch_size)
        assert batch.batch_size == batch_size

    def test_convert_pred_entity_to_compute_metric(self, model: OTX3DDetectionModel, batch_data_entity):
        model.training = False
        outputs = {
            "scores": torch.randn(2, 50, 2),
            "boxes_3d": torch.randn(2, 50, 6),
            "boxes": torch.randn(2, 50, 4),
            "size_3d": torch.randn(2, 50, 3),
            "depth": torch.randn(2, 50, 2),
            "heading_angle": torch.randn(2, 50, 24),
        }
        customized_outputs = model._customize_outputs(outputs, batch_data_entity)
        converted_pred = model._convert_pred_entity_to_compute_metric(customized_outputs, batch_data_entity)

        assert "preds" in converted_pred
        assert "target" in converted_pred
