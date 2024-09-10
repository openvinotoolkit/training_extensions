# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for keypoint detection model entity."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from otx.algo.keypoint_detection.rtmpose import RTMPoseTiny
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.keypoint_detection import KeypointDetBatchDataEntity, KeypointDetBatchPredEntity
from otx.core.metrics.pck import PCKMeasureCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.types.label import LabelInfo

if TYPE_CHECKING:
    from otx.core.model.keypoint_detection import OTXKeypointDetectionModel


class TestOTXKeypointDetectionModel:
    @pytest.fixture()
    def model(self, label_info, optimizer, scheduler, metric, torch_compile) -> OTXKeypointDetectionModel:
        return RTMPoseTiny(label_info, (512, 512), optimizer, scheduler, metric, torch_compile)

    @pytest.fixture()
    def batch_data_entity(self, model) -> KeypointDetBatchDataEntity:
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
        return PCKMeasureCallable

    @pytest.fixture()
    def torch_compile(self):
        return False

    def test_export_parameters(self, model):
        params = model._export_parameters
        assert params.model_type == "keypoint_detection"
        assert params.task_type == "keypoint_detection"

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
        assert customized_inputs["inputs"].shape == (2, 3, model.input_size[0], model.input_size[1])
        assert "mode" in customized_inputs

    def test_customize_outputs_training(self, model, batch_data_entity):
        outputs = {"loss": torch.tensor(0.5)}
        customized_outputs = model._customize_outputs(outputs, batch_data_entity)
        assert isinstance(customized_outputs, OTXBatchLossEntity)
        assert customized_outputs["loss"] == torch.tensor(0.5)

    def test_customize_outputs_predict(self, model, batch_data_entity):
        model.training = False
        outputs = [(torch.randn(2, 2, 2), torch.randn(2, 2, 2))]
        customized_outputs = model._customize_outputs(outputs, batch_data_entity)
        assert isinstance(customized_outputs, KeypointDetBatchPredEntity)
        assert len(customized_outputs.keypoints) == len(customized_outputs.scores)

    def test_dummy_input(self, model: OTXKeypointDetectionModel):
        batch_size = 2
        batch = model.get_dummy_input(batch_size)
        assert batch.batch_size == batch_size
