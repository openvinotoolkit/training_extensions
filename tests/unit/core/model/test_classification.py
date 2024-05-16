# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for classification model module."""

from __future__ import annotations

from unittest.mock import create_autospec

import pytest
import torch
from lightning.pytorch.cli import ReduceLROnPlateau
from omegaconf.dictconfig import DictConfig
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.classification import (
    MulticlassClsBatchPredEntity,
    MultilabelClsBatchPredEntity,
)
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.model.classification import (
    MMPretrainMulticlassClsModel,
    MMPretrainMultilabelClsModel,
    OTXHlabelClsModel,
    OTXMulticlassClsModel,
    OTXMultilabelClsModel,
)
from otx.core.types.export import TaskLevelExportParameters
from torch.optim import Optimizer


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
            torch_compile=False,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
        )

        assert isinstance(model._export_parameters, TaskLevelExportParameters)
        assert model._export_parameters.model_type.lower() == "classification"
        assert model._export_parameters.task_type.lower() == "classification"
        assert not model._export_parameters.multilabel
        assert not model._export_parameters.hierarchical

        model = OTXMultilabelClsModel(
            label_info=1,
            torch_compile=False,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
        )

        assert model._export_parameters.multilabel
        assert not model._export_parameters.hierarchical

        model = OTXHlabelClsModel(
            label_info=fxt_hlabel_multilabel_info,
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


class TestMMPretrainMulticlassClsModel:
    @pytest.fixture()
    def mock_optimizer(self):
        return lambda _: create_autospec(Optimizer)

    @pytest.fixture()
    def mock_scheduler(self):
        return lambda _: create_autospec([ReduceLROnPlateau])

    @pytest.fixture(
        params=[
            {
                "state_dict": {},
            },
            {
                "hyper_parameters": {},
                "state_dict": {},
            },
        ],
        ids=["v1", "v2"],
    )
    def mock_ckpt(self, request):
        return request.param

    @pytest.fixture()
    def config(self) -> DictConfig:
        cfg_dict = {
            "type": "ImageClassifier",
            "backbone": {"type": "MobileNetV3", "arch": "large"},
            "neck": {"type": "GlobalAveragePooling"},
            "head": {
                "type": "StackedLinearClsHead",
                "num_classes": 1000,
                "in_channels": 960,
                "mid_channels": [1280],
                "dropout_rate": 0.2,
                "act_cfg": {"type": "HSwish"},
                "loss": {"type": "CrossEntropyLoss", "loss_weight": 1.0},
                "init_cfg": {"type": "Normal", "layer": "Linear", "mean": 0.0, "std": 0.01, "bias": 0.0},
                "topk": (1, 5),
            },
            "data_preprocessor": {
                "type": "ClsDataPreprocessor",
                "mean": [123.675, 116.28, 103.53],
                "std": [58.395, 57.12, 57.375],
            },
        }
        return DictConfig(cfg_dict)

    @pytest.fixture()
    def otx_model(
        self,
        mock_optimizer,
        mock_scheduler,
        config,
    ) -> MMPretrainMulticlassClsModel:
        return MMPretrainMulticlassClsModel(
            label_info=1,
            config=config,
            torch_compile=False,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
        )

    def test_configure_metric_with_ckpt(
        self,
        otx_model,
        mock_ckpt,
    ) -> None:
        otx_model.on_load_checkpoint(mock_ckpt)

    def test_create_model(self, otx_model) -> None:
        mmpretrain_model = otx_model._create_model()
        assert mmpretrain_model is not None
        assert isinstance(mmpretrain_model, torch.nn.Module)

    def test_customize_inputs(self, otx_model, fxt_multi_class_cls_data_entity) -> None:
        output_data = otx_model._customize_inputs(fxt_multi_class_cls_data_entity[2])
        assert isinstance(output_data, dict)
        assert "mode" in output_data
        assert output_data["mode"] == "loss"
        assert "inputs" in output_data
        assert "data_samples" in output_data

    def test_customize_outputs(self, otx_model, fxt_multi_class_cls_data_entity):
        customized_input = otx_model._customize_inputs(fxt_multi_class_cls_data_entity[2])
        outputs = otx_model.model(**customized_input)
        otx_model.training = True
        preds = otx_model._customize_outputs(outputs, fxt_multi_class_cls_data_entity[2])
        assert isinstance(preds, OTXBatchLossEntity)

        # With wrong outputs (Not Dict type)
        wrong_outputs = torch.Tensor([1, 2, 3])
        with pytest.raises(TypeError):
            otx_model._customize_outputs(wrong_outputs, fxt_multi_class_cls_data_entity[2])

        otx_model.training = False
        customized_input["mode"] = "predict"
        outputs = otx_model.model(**customized_input)
        preds = otx_model._customize_outputs(outputs, fxt_multi_class_cls_data_entity[2])
        assert isinstance(preds, MulticlassClsBatchPredEntity)

        # Insert wrong outputs (Not DataSample)
        wrong_outputs = [torch.Tensor([1, 2, 3])]
        with pytest.raises(TypeError):
            otx_model._customize_outputs(wrong_outputs, fxt_multi_class_cls_data_entity[2])

        # Explain Mode
        otx_model.explain_mode = True
        with pytest.raises(ValueError, match="Model output should be a dict"):
            otx_model._customize_outputs(outputs, fxt_multi_class_cls_data_entity[2])

        # Without feature_vector
        explain_outputs = {"logits": outputs, "saliency_map": torch.Tensor([1, 2, 3])}
        with pytest.raises(ValueError, match="No feature vector"):
            otx_model._customize_outputs(explain_outputs, fxt_multi_class_cls_data_entity[2])

        # Without saliency_map
        explain_outputs = {"logits": outputs, "feature_vector": torch.Tensor([1, 2, 3])}
        with pytest.raises(ValueError, match="No saliency maps"):
            otx_model._customize_outputs(explain_outputs, fxt_multi_class_cls_data_entity[2])

        explain_outputs = {
            "logits": outputs,
            "feature_vector": torch.Tensor([1, 2, 3]),
            "saliency_map": torch.Tensor([1, 2, 3]),
        }
        preds = otx_model._customize_outputs(explain_outputs, fxt_multi_class_cls_data_entity[2])
        assert isinstance(preds, MulticlassClsBatchPredEntity)

    def test_export_parameters(self, otx_model):
        parameters = otx_model._export_parameters
        assert isinstance(parameters, TaskLevelExportParameters)
        assert parameters.task_type == "classification"

    def test_exporter(self, otx_model):
        exporter = otx_model._exporter
        assert isinstance(exporter, OTXNativeModelExporter)

    def test_forward_for_tracing(self, otx_model):
        otx_model.eval()
        output = otx_model.forward_for_tracing(torch.randn(1, 3, 32, 32))
        assert len(output) == 1


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


class TestMMPretrainMultilabelClsModel:
    @pytest.fixture()
    def mock_optimizer(self):
        return lambda _: create_autospec(Optimizer)

    @pytest.fixture()
    def mock_scheduler(self):
        return lambda _: create_autospec([ReduceLROnPlateau])

    @pytest.fixture(
        params=[
            {
                "state_dict": {},
            },
            {
                "hyper_parameters": {},
                "state_dict": {},
            },
        ],
        ids=["v1", "v2"],
    )
    def mock_ckpt(self, request):
        return request.param

    @pytest.fixture()
    def config(self) -> DictConfig:
        cfg_dict = {
            "type": "ImageClassifier",
            "backbone": {"type": "MobileNetV3", "arch": "large"},
            "neck": {"type": "GlobalAveragePooling"},
            "head": {
                "type": "MultiLabelLinearClsHead",
                "num_classes": 1000,
                "in_channels": 960,
                "loss": {"type": "CrossEntropyLoss", "loss_weight": 1.0},
                "init_cfg": {"type": "Normal", "layer": "Linear", "mean": 0.0, "std": 0.01, "bias": 0.0},
                "topk": 1,
            },
            "data_preprocessor": {
                "type": "ClsDataPreprocessor",
                "mean": [123.675, 116.28, 103.53],
                "std": [58.395, 57.12, 57.375],
            },
        }
        return DictConfig(cfg_dict)

    @pytest.fixture()
    def otx_model(
        self,
        mock_optimizer,
        mock_scheduler,
        config,
    ) -> MMPretrainMultilabelClsModel:
        return MMPretrainMultilabelClsModel(
            label_info=1,
            config=config,
            torch_compile=False,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
        )

    def test_configure_metric_with_ckpt(
        self,
        otx_model,
        mock_ckpt,
    ) -> None:
        otx_model.on_load_checkpoint(mock_ckpt)

    def test_create_model(self, otx_model) -> None:
        mmpretrain_model = otx_model._create_model()
        assert mmpretrain_model is not None
        assert isinstance(mmpretrain_model, torch.nn.Module)

    def test_customize_inputs(self, otx_model, fxt_multi_label_cls_data_entity) -> None:
        output_data = otx_model._customize_inputs(fxt_multi_label_cls_data_entity[2])
        assert isinstance(output_data, dict)
        assert "mode" in output_data
        assert output_data["mode"] == "loss"
        assert "inputs" in output_data
        assert "data_samples" in output_data

    def test_customize_outputs(self, otx_model, fxt_multi_label_cls_data_entity):
        customized_input = otx_model._customize_inputs(fxt_multi_label_cls_data_entity[2])
        outputs = otx_model.model(**customized_input)
        otx_model.training = True
        preds = otx_model._customize_outputs(outputs, fxt_multi_label_cls_data_entity[2])
        assert isinstance(preds, OTXBatchLossEntity)

        # With wrong outputs (Not Dict type)
        wrong_outputs = torch.Tensor([1, 2, 3])
        with pytest.raises(TypeError):
            otx_model._customize_outputs(wrong_outputs, fxt_multi_label_cls_data_entity[2])

        otx_model.training = False
        customized_input["mode"] = "predict"
        outputs = otx_model.model(**customized_input)
        preds = otx_model._customize_outputs(outputs, fxt_multi_label_cls_data_entity[2])
        assert isinstance(preds, MultilabelClsBatchPredEntity)

        # Insert wrong outputs (Not DataSample)
        wrong_outputs = [torch.Tensor([1, 2, 3])]
        with pytest.raises(TypeError):
            otx_model._customize_outputs(wrong_outputs, fxt_multi_label_cls_data_entity[2])

        # Explain Mode
        otx_model.explain_mode = True
        with pytest.raises(ValueError, match="Model output should be a dict"):
            otx_model._customize_outputs(outputs, fxt_multi_label_cls_data_entity[2])

        # Without feature_vector
        explain_outputs = {"logits": outputs, "saliency_map": torch.Tensor([1, 2, 3])}
        with pytest.raises(ValueError, match="No feature vector"):
            otx_model._customize_outputs(explain_outputs, fxt_multi_label_cls_data_entity[2])

        # Without saliency_map
        explain_outputs = {"logits": outputs, "feature_vector": torch.Tensor([1, 2, 3])}
        with pytest.raises(ValueError, match="No saliency maps"):
            otx_model._customize_outputs(explain_outputs, fxt_multi_label_cls_data_entity[2])

        explain_outputs = {
            "logits": outputs,
            "feature_vector": torch.Tensor([1, 2, 3]),
            "saliency_map": torch.Tensor([1, 2, 3]),
        }
        preds = otx_model._customize_outputs(explain_outputs, fxt_multi_label_cls_data_entity[2])
        assert isinstance(preds, MultilabelClsBatchPredEntity)

    def test_export_parameters(self, otx_model):
        parameters = otx_model._export_parameters
        assert isinstance(parameters, TaskLevelExportParameters)
        assert parameters.model_type.lower() == "classification"
        assert parameters.task_type.lower() == "classification"
        assert parameters.multilabel
        assert not parameters.hierarchical

    def test_exporter(self, otx_model):
        exporter = otx_model._exporter
        assert isinstance(exporter, OTXNativeModelExporter)


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
        model = OTXHlabelClsModel(label_info=fxt_hlabel_multilabel_info)
        assert model.label_info.num_multilabel_classes == fxt_hlabel_multilabel_info.num_multilabel_classes

        fxt_hlabel_multilabel_info.num_multilabel_classes = 0
        model.label_info = fxt_hlabel_multilabel_info
        assert model.label_info.num_multilabel_classes == 0
