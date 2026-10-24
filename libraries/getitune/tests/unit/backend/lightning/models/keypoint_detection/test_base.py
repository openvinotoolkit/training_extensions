# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for keypoint detection model entity."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from getitune.backend.lightning.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from getitune.backend.lightning.models.keypoint_detection.rtmpose import RTMPose
from getitune.data.entity.base import BatchLoss
from getitune.data.entity.sample import PredictionBatch, SampleBatch
from getitune.metrics.pck import PCKMeasureCallable
from getitune.types.label import LabelInfo

if TYPE_CHECKING:
    from getitune.backend.lightning.models.keypoint_detection.base import LightningKeypointDetectionModel


class TestLightningKeypointDetectionModel:
    @pytest.fixture
    def model(self, label_info, optimizer, scheduler, metric, torch_compile) -> LightningKeypointDetectionModel:
        return RTMPose(
            label_info=label_info,
            data_input_params=DataInputParams((512, 512), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    @pytest.fixture
    def batch_data_entity(self, model) -> SampleBatch:
        return model.get_dummy_input(2)

    @pytest.fixture
    def label_info(self) -> LabelInfo:
        return LabelInfo(
            label_names=["label_0", "label_1"],
            label_groups=[["label_0", "label_1"]],
            label_ids=["0", "1"],
        )

    @pytest.fixture
    def optimizer(self):
        return DefaultOptimizerCallable

    @pytest.fixture
    def scheduler(self):
        return DefaultSchedulerCallable

    @pytest.fixture
    def metric(self):
        return PCKMeasureCallable

    @pytest.fixture
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
                LabelInfo(
                    label_names=["label1", "label2", "label3"],
                    label_groups=[["label1", "label2", "label3"]],
                    label_ids=["0", "1", "2"],
                ),
                LabelInfo(
                    label_names=["label1", "label2", "label3"],
                    label_groups=[["label1", "label2", "label3"]],
                    label_ids=["0", "1", "2"],
                ),
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
        assert customized_inputs["inputs"].shape == (2, 3, *model.data_input_params.input_size)
        assert "mode" in customized_inputs

    def test_customize_outputs_training(self, model, batch_data_entity):
        outputs = {"loss": torch.tensor(0.5)}
        customized_outputs = model._customize_outputs(outputs, batch_data_entity)
        assert isinstance(customized_outputs, BatchLoss)
        assert customized_outputs["loss"] == torch.tensor(0.5)

    def test_customize_outputs_predict(self, model, batch_data_entity):
        model.training = False
        outputs = [(torch.randn(2, 2), torch.randn(2))]
        customized_outputs = model._customize_outputs(outputs, batch_data_entity)
        assert isinstance(customized_outputs, PredictionBatch)
        assert len(customized_outputs.keypoints) == len(customized_outputs.scores)

    def test_dummy_input(self, model: LightningKeypointDetectionModel):
        batch_size = 2
        batch = model.get_dummy_input(batch_size)
        assert batch.batch_size == batch_size
