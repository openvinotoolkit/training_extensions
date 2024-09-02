# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for classification model module."""

from __future__ import annotations

from unittest.mock import create_autospec

import pytest
from lightning.pytorch.cli import ReduceLROnPlateau
from otx.core.model.classification import OTXHlabelClsModel, OTXMulticlassClsModel, OTXMultilabelClsModel
from otx.core.types.export import TaskLevelExportParameters
from torch.optim import Optimizer

SKIP_MMLAB_TEST = False
try:
    import mmpretrain  # noqa: F401
except ImportError:
    SKIP_MMLAB_TEST = True


class TestOTXMulticlassClsModel:
    @pytest.fixture()
    def mock_optimizer(self):
        return lambda _: create_autospec(Optimizer)

    @pytest.fixture()
    def mock_scheduler(self):
        return lambda _: create_autospec([ReduceLROnPlateau])

    def test_export_parameters(
        self,
        mock_optimizer,
        mock_scheduler,
        fxt_hlabel_multilabel_info,
    ) -> None:
        model = OTXMulticlassClsModel(
            label_info=1,
            input_size=(224, 224),
            torch_compile=False,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
        )

        assert isinstance(model._export_parameters, TaskLevelExportParameters)
        assert model._export_parameters.model_type.lower() == "classification"
        assert model._export_parameters.task_type.lower() == "classification"
        assert not model._export_parameters.multilabel
        assert not model._export_parameters.hierarchical
        assert model._export_parameters.output_raw_scores

        model = OTXMultilabelClsModel(
            label_info=1,
            input_size=(224, 224),
            torch_compile=False,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
        )

        assert model._export_parameters.multilabel
        assert not model._export_parameters.hierarchical

        model = OTXHlabelClsModel(
            label_info=fxt_hlabel_multilabel_info,
            input_size=(224, 224),
            torch_compile=False,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
        )

        assert not model._export_parameters.multilabel
        assert model._export_parameters.hierarchical

    def test_convert_pred_entity_to_compute_metric(
        self,
        mock_optimizer,
        mock_scheduler,
        fxt_multi_class_cls_data_entity,
    ) -> None:
        model = OTXMulticlassClsModel(
            label_info=1,
            input_size=(224, 224),
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


class TestOTXMultilabelClsModel:
    @pytest.fixture()
    def mock_optimizer(self):
        return lambda _: create_autospec(Optimizer)

    @pytest.fixture()
    def mock_scheduler(self):
        return lambda _: create_autospec([ReduceLROnPlateau])

    def test_export_parameters(
        self,
        mock_optimizer,
        mock_scheduler,
    ) -> None:
        model = OTXMultilabelClsModel(
            label_info=1,
            input_size=(224, 224),
            torch_compile=False,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
        )

        assert isinstance(model._export_parameters, TaskLevelExportParameters)
        assert model._export_parameters.model_type.lower() == "classification"
        assert model._export_parameters.task_type.lower() == "classification"
        assert model._export_parameters.multilabel
        assert not model._export_parameters.hierarchical

    def test_convert_pred_entity_to_compute_metric(
        self,
        mock_optimizer,
        mock_scheduler,
        fxt_multi_label_cls_data_entity,
    ) -> None:
        model = OTXMultilabelClsModel(
            label_info=1,
            input_size=(224, 224),
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


class TestOTXHlabelClsModel:
    @pytest.fixture()
    def mock_optimizer(self):
        return lambda _: create_autospec(Optimizer)

    @pytest.fixture()
    def mock_scheduler(self):
        return lambda _: create_autospec([ReduceLROnPlateau])

    def test_export_parameters(
        self,
        mock_optimizer,
        mock_scheduler,
        fxt_hlabel_multilabel_info,
    ) -> None:
        model = OTXHlabelClsModel(
            label_info=fxt_hlabel_multilabel_info,
            input_size=(224, 224),
            torch_compile=False,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
        )

        assert isinstance(model._export_parameters, TaskLevelExportParameters)
        assert model._export_parameters.model_type.lower() == "classification"
        assert model._export_parameters.task_type.lower() == "classification"
        assert not model._export_parameters.multilabel
        assert model._export_parameters.hierarchical

    def test_convert_pred_entity_to_compute_metric(
        self,
        mock_optimizer,
        mock_scheduler,
        fxt_h_label_cls_data_entity,
        fxt_hlabel_multilabel_info,
    ) -> None:
        model = OTXHlabelClsModel(
            label_info=fxt_hlabel_multilabel_info,
            input_size=(224, 224),
            torch_compile=False,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
        )
        metric_input = model._convert_pred_entity_to_compute_metric(
            fxt_h_label_cls_data_entity[1],
            fxt_h_label_cls_data_entity[2],
        )

        assert isinstance(metric_input, dict)
        assert "preds" in metric_input
        assert "target" in metric_input

        model.label_info.num_multilabel_classes = 0
        metric_input = model._convert_pred_entity_to_compute_metric(
            fxt_h_label_cls_data_entity[1],
            fxt_h_label_cls_data_entity[2],
        )
        assert isinstance(metric_input, dict)
        assert "preds" in metric_input
        assert "target" in metric_input

    def test_set_label_info(self, fxt_hlabel_multilabel_info):
        model = OTXHlabelClsModel(label_info=fxt_hlabel_multilabel_info, input_size=(224, 224))
        assert model.label_info.num_multilabel_classes == fxt_hlabel_multilabel_info.num_multilabel_classes

        fxt_hlabel_multilabel_info.num_multilabel_classes = 0
        model.label_info = fxt_hlabel_multilabel_info
        assert model.label_info.num_multilabel_classes == 0
