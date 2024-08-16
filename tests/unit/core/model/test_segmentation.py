# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for segmentation model entity."""

from __future__ import annotations

import pytest
import torch
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.segmentation import SegBatchDataEntity, SegBatchPredEntity
from otx.core.metrics.dice import SegmCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.segmentation import OTXSegmentationModel
from otx.core.types.label import SegLabelInfo


class TestOTXSegmentationModel:
    @pytest.fixture()
    def model(self, label_info, optimizer, scheduler, metric, torch_compile):
        return OTXSegmentationModel(label_info, (512, 512), optimizer, scheduler, metric, torch_compile)

    @pytest.fixture()
    def batch_data_entity(self):
        return SegBatchDataEntity(
            batch_size=2,
            images=torch.randn(2, 3, 224, 224),
            imgs_info=[],
            masks=[torch.randn(224, 224), torch.randn(224, 224)],
        )

    @pytest.fixture()
    def label_info(self):
        return SegLabelInfo(
            label_names=["Background", "label_0", "label_1"],
            label_groups=[["Background", "label_0", "label_1"]],
        )

    @pytest.fixture()
    def optimizer(self):
        return DefaultOptimizerCallable

    @pytest.fixture()
    def scheduler(self):
        return DefaultSchedulerCallable

    @pytest.fixture()
    def metric(self):
        return SegmCallable

    @pytest.fixture()
    def torch_compile(self):
        return False

    def test_export_parameters(self, model):
        params = model._export_parameters
        assert params.model_type == "Segmentation"
        assert params.task_type == "segmentation"
        assert params.return_soft_prediction is True
        assert params.soft_threshold == 0.5
        assert params.blur_strength == -1

    @pytest.mark.parametrize(
        ("label_info", "expected_label_info"),
        [
            (
                SegLabelInfo(label_names=["label1", "label2", "label3"], label_groups=[["label1", "label2", "label3"]]),
                SegLabelInfo(label_names=["label1", "label2", "label3"], label_groups=[["label1", "label2", "label3"]]),
            ),
            (SegLabelInfo.from_num_classes(num_classes=5), SegLabelInfo.from_num_classes(num_classes=5)),
        ],
    )
    def test_dispatch_label_info(self, model, label_info, expected_label_info):
        result = model._dispatch_label_info(label_info)
        assert result == expected_label_info

    def test_init(self, model):
        assert model.num_classes == 3

    def test_customize_inputs(self, model, batch_data_entity):
        customized_inputs = model._customize_inputs(batch_data_entity)
        assert customized_inputs["inputs"].shape == (2, 3, 224, 224)
        assert customized_inputs["img_metas"] == []
        assert customized_inputs["masks"].shape == (2, 224, 224)

    def test_customize_outputs_training(self, model, batch_data_entity):
        outputs = {"loss": torch.tensor(0.5)}
        customized_outputs = model._customize_outputs(outputs, batch_data_entity)
        assert isinstance(customized_outputs, OTXBatchLossEntity)
        assert customized_outputs["loss"] == torch.tensor(0.5)

    def test_customize_outputs_predict(self, model, batch_data_entity):
        model.training = False
        outputs = torch.randn(2, 10, 224, 224)
        customized_outputs = model._customize_outputs(outputs, batch_data_entity)
        assert isinstance(customized_outputs, SegBatchPredEntity)
        assert len(customized_outputs.scores) == 0
        assert customized_outputs.images.shape == (2, 3, 224, 224)
        assert customized_outputs.imgs_info == []

    def test_dummy_input(self, model: OTXSegmentationModel):
        batch_size = 2
        batch = model.get_dummy_input(batch_size)
        assert batch.batch_size == batch_size
