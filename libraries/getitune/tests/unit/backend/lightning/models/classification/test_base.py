# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for classification model module."""

from __future__ import annotations

from unittest.mock import create_autospec

import pytest
from lightning.pytorch.cli import ReduceLROnPlateau
from torch import nn
from torch.optim import Optimizer

from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.lightning.models.classification.multiclass_models.base import LightningMulticlassClsModel
from getitune.backend.lightning.models.classification.multilabel_models.base import LightningMultilabelClsModel
from getitune.types.export import TaskLevelExportParameters


class MockClsModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.backbone = nn.Sequential()
        self.head = nn.Linear(5, 2)

    def init_weights(self):
        pass


class TestLightningMulticlassClsModel:
    @pytest.fixture(autouse=True)
    def mock_model(self, mocker):
        LightningMulticlassClsModel._build_model = mocker.MagicMock(return_value=MockClsModel())

    @pytest.fixture
    def mock_optimizer(self):
        return lambda _: create_autospec(Optimizer)

    @pytest.fixture
    def mock_scheduler(self):
        return lambda _: create_autospec([ReduceLROnPlateau])

    def test_export_parameters(
        self,
        mock_optimizer,
        mock_scheduler,
    ) -> None:
        model = LightningMulticlassClsModel(
            label_info=1,
            data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            torch_compile=False,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
        )

        assert isinstance(model._export_parameters, TaskLevelExportParameters)
        assert model._export_parameters.model_type.lower() == "classification"
        assert model._export_parameters.task_type.lower() == "classification"
        assert not model._export_parameters.multilabel
        assert model._export_parameters.output_raw_scores

    def test_convert_pred_entity_to_compute_metric(
        self,
        mock_optimizer,
        mock_scheduler,
        fxt_multi_class_cls_data_entity,
    ) -> None:
        model = LightningMulticlassClsModel(
            label_info=1,
            data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            torch_compile=False,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
        )
        metric_input = model._convert_pred_entity_to_compute_metric(
            fxt_multi_class_cls_data_entity[1],
            fxt_multi_class_cls_data_entity[2],
        )

        assert isinstance(metric_input, dict)
        assert "preds" in metric_input
        assert "target" in metric_input


class TestLightningMultilabelClsModel:
    @pytest.fixture(autouse=True)
    def mock_model(self, mocker):
        LightningMultilabelClsModel._build_model = mocker.MagicMock(return_value=MockClsModel())

    @pytest.fixture
    def mock_optimizer(self):
        return lambda _: create_autospec(Optimizer)

    @pytest.fixture
    def mock_scheduler(self):
        return lambda _: create_autospec([ReduceLROnPlateau])

    def test_export_parameters(
        self,
        mock_optimizer,
        mock_scheduler,
    ) -> None:
        model = LightningMultilabelClsModel(
            label_info=1,
            data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            torch_compile=False,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
        )

        assert isinstance(model._export_parameters, TaskLevelExportParameters)
        assert model._export_parameters.model_type.lower() == "classification"
        assert model._export_parameters.task_type.lower() == "classification"
        assert model._export_parameters.multilabel

    def test_convert_pred_entity_to_compute_metric(
        self,
        mock_optimizer,
        mock_scheduler,
        fxt_multi_label_cls_data_entity,
    ) -> None:
        model = LightningMultilabelClsModel(
            label_info=1,
            data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            torch_compile=False,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
        )
        metric_input = model._convert_pred_entity_to_compute_metric(
            fxt_multi_label_cls_data_entity[1],
            fxt_multi_label_cls_data_entity[2],
        )

        assert isinstance(metric_input, dict)
        assert "preds" in metric_input
        assert "target" in metric_input
