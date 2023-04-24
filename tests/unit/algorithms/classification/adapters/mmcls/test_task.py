"""Unit Test for otx.algorithms.detection.adapters.mmdet.task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import json
from contextlib import nullcontext
from copy import deepcopy
from typing import Any, Dict

import numpy as np
import pytest
import torch
from mmcv.utils import Config
from torch import nn

from otx.algorithms.common.adapters.mmcv.utils.config_utils import MPAConfig
from otx.algorithms.classification.adapters.mmcls.task import MMClassificationTask
from otx.algorithms.classification.adapters.mmcls.models.classifiers.sam_classifier import SAMImageClassifier
from otx.algorithms.classification.configs.base import ClassificationConfig
from otx.api.configuration import ConfigurableParameters
from otx.api.configuration.helper import create
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.explain_parameters import ExplainParameters
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.label import Domain
from otx.api.entities.label_schema import LabelGroup, LabelGroupType, LabelSchemaEntity
from otx.api.entities.model import (
    ModelConfiguration,
    ModelEntity,
    ModelFormat,
    ModelOptimizationType,
    ModelPrecision,
)
from otx.api.entities.model_template import InstantiationType, parse_model_template, TaskFamily, TaskType
from otx.api.entities.resultset import ResultSetEntity
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.classification.test_helper import (
    DEFAULT_CLS_TEMPLATE_DIR,
    init_environment,
    generate_cls_dataset,
    generate_label_schema,
)


class MockModule(nn.Module):
    """Mock class for nn.Module."""

    def forward(self, inputs: Any):
        return inputs


class MockModel(nn.Module):
    """Mock class for pytorch model."""

    def __init__(self):
        super().__init__()
        self.module = MockModule()
        self.module.backbone = MockModule()
        self.backbone = MockModule()

    def forward(self, *args, **kwargs):
        forward_hooks = list(self.module.backbone._forward_hooks.values())
        for hook in forward_hooks:
            hook(1, 2, 3)
        return np.array([[0.3, 0.7]])

    @staticmethod
    def named_parameters(*args, **kwargs):
        return {"name": torch.Tensor([0.5])}.items()


class MockDataset(DatasetEntity):
    """Mock class for mm_dataset."""

    def __init__(self, dataset: DatasetEntity):
        self.dataset = dataset
        self.CLASSES = ["1", "2", "3"]

    def __len__(self):
        return len(self.dataset)

    def evaluate(self, prediction, *args, **kwargs):
        return {"mAP": 1.0}


class MockDataLoader:
    """Mock class for data loader."""

    def __init__(self, dataset: DatasetEntity):
        self.dataset = dataset
        self.iter = iter(self.dataset)

    def __len__(self) -> int:
        return len(self.dataset)

    def __next__(self) -> Dict[str, DatasetItemEntity]:
        return {"imgs": next(self.iter)}

    def __iter__(self):
        return self


class MockExporter:
    """Mock class for Exporter."""

    def __init__(self, task):
        self._output_path = task._output_path

    def run(self, *args, **kwargs):
        with open(os.path.join(self._output_path, "openvino.bin"), "wb") as f:
            f.write(np.ndarray([0]))
        with open(os.path.join(self._output_path, "openvino.xml"), "wb") as f:
            f.write(np.ndarray([0]))
        with open(os.path.join(self._output_path, "model.onnx"), "wb") as f:
            f.write(np.ndarray([0]))

        return {
            "outputs": {
                "bin": os.path.join(self._output_path, "openvino.bin"),
                "xml": os.path.join(self._output_path, "openvino.xml"),
                "onnx": os.path.join(self._output_path, "model.onnx"),
            }
        }


class TestMMClassificationTask:
    """Test class for MMClassificationTask.

    Details are explained in each test function.
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        model_template = parse_model_template(os.path.join(DEFAULT_CLS_TEMPLATE_DIR, "template.yaml"))
        hyper_parameters = create(model_template.hyper_parameters.data)

        mc_task_env, self.mc_cls_dataset = init_environment(hyper_parameters, model_template, False, False, 100)
        self.mc_cls_task = MMClassificationTask(mc_task_env)
        self.mc_cls_label_schema = generate_label_schema(self.mc_cls_dataset.get_labels(), False, False)

        ml_task_env, self.ml_cls_dataset = init_environment(hyper_parameters, model_template, True, False, 100)
        self.ml_cls_task = MMClassificationTask(ml_task_env)
        self.ml_cls_label_schema = generate_label_schema(self.ml_cls_dataset.get_labels(), False, False)

        hl_task_env, self.hl_cls_dataset = init_environment(hyper_parameters, model_template, False, True, 100)
        self.hl_cls_task = MMClassificationTask(hl_task_env)
        self.hl_cls_label_schema = generate_label_schema(self.hl_cls_dataset.get_labels(), False, False)

    @e2e_pytest_unit
    def test_build_model(self, mocker) -> None:
        """Test build_model function."""
        _mock_recipe_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_CLS_TEMPLATE_DIR, "model.py"))
        _mock_recipe_cfg.model.pop("task")
        model = self.mc_cls_task.build_model(_mock_recipe_cfg, True)
        assert isinstance(model, SAMImageClassifier)

    @e2e_pytest_unit
    def test_train_multiclass(self, mocker) -> None:
        """Test train function."""

        def _mock_train_model(*args, **kwargs):
            with open(os.path.join(self.mc_cls_task._output_path, "latest.pth"), "wb") as f:
                torch.save({"dummy": torch.randn(1, 3, 3, 3)}, f)

        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.task.build_dataset",
            return_value=MockDataset(self.mc_cls_dataset),
        )
        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.task.build_dataloader",
            return_value=MockDataLoader(self.mc_cls_dataset),
        )
        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.task.train_model",
            side_effect=_mock_train_model,
        )
        mocker.patch.object(MMClassificationTask, "build_model", return_value=MockModel())
        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.task.build_data_parallel",
            return_value=MockModel(),
        )
        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.task.FeatureVectorHook",
            return_value=nullcontext(),
        )

        _config = ModelConfiguration(ClassificationConfig("header"), self.mc_cls_label_schema)
        output_model = ModelEntity(self.mc_cls_dataset, _config)
        self.mc_cls_task.train(self.mc_cls_dataset, output_model)
        output_model.performance == 1.0

    @e2e_pytest_unit
    def test_train_multilabel(self, mocker) -> None:
        """Test train function."""

        def _mock_train_model(*args, **kwargs):
            with open(os.path.join(self.ml_cls_task._output_path, "latest.pth"), "wb") as f:
                torch.save({"dummy": torch.randn(1, 3, 3, 3)}, f)

        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.task.build_dataset",
            return_value=MockDataset(self.ml_cls_dataset),
        )
        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.task.build_dataloader",
            return_value=MockDataLoader(self.ml_cls_dataset),
        )
        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.task.train_model",
            side_effect=_mock_train_model,
        )
        mocker.patch.object(MMClassificationTask, "build_model", return_value=MockModel())
        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.task.build_data_parallel",
            return_value=MockModel(),
        )
        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.task.FeatureVectorHook",
            return_value=nullcontext(),
        )

        _config = ModelConfiguration(ClassificationConfig("header"), self.ml_cls_label_schema)
        output_model = ModelEntity(self.ml_cls_dataset, _config)
        self.ml_cls_task.train(self.ml_cls_dataset, output_model)
        output_model.performance == 1.0

    @e2e_pytest_unit
    def test_train_hierarchicallabel(self, mocker) -> None:
        """Test train function."""

        def _mock_train_model(*args, **kwargs):
            with open(os.path.join(self.hl_cls_task._output_path, "latest.pth"), "wb") as f:
                torch.save({"dummy": torch.randn(1, 3, 3, 3)}, f)

        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.task.build_dataset",
            return_value=MockDataset(self.hl_cls_dataset),
        )
        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.task.build_dataloader",
            return_value=MockDataLoader(self.hl_cls_dataset),
        )
        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.task.train_model",
            side_effect=_mock_train_model,
        )
        mocker.patch.object(MMClassificationTask, "build_model", return_value=MockModel())
        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.task.build_data_parallel",
            return_value=MockModel(),
        )
        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.task.FeatureVectorHook",
            return_value=nullcontext(),
        )

        _config = ModelConfiguration(ClassificationConfig("header"), self.hl_cls_label_schema)
        output_model = ModelEntity(self.hl_cls_dataset, _config)
        self.hl_cls_task.train(self.hl_cls_dataset, output_model)
        output_model.performance == 1.0

    @e2e_pytest_unit
    def test_infer_multiclass(self, mocker) -> None:
        """Test infer function."""

        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.task.build_dataset",
            return_value=MockDataset(self.mc_cls_dataset),
        )
        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.task.build_dataloader",
            return_value=MockDataLoader(self.mc_cls_dataset),
        )
        mocker.patch.object(MMClassificationTask, "build_model", return_value=MockModel())
        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.task.build_data_parallel",
            return_value=MockModel(),
        )
        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.task.FeatureVectorHook",
            return_value=nullcontext(),
        )

        inference_parameters = InferenceParameters(is_evaluation=True)
        outputs = self.mc_cls_task.infer(self.mc_cls_dataset.with_empty_annotations(), inference_parameters)
        for output in outputs:
            assert output.get_annotations()[-1].get_labels()[0].probability == 0.7

    @e2e_pytest_unit
    def test_infer_multilabel(self, mocker) -> None:
        """Test infer function."""

        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.task.build_dataset",
            return_value=MockDataset(self.ml_cls_dataset),
        )
        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.task.build_dataloader",
            return_value=MockDataLoader(self.ml_cls_dataset),
        )
        mocker.patch.object(MMClassificationTask, "build_model", return_value=MockModel())
        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.task.build_data_parallel",
            return_value=MockModel(),
        )
        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.task.FeatureVectorHook",
            return_value=nullcontext(),
        )

        inference_parameters = InferenceParameters(is_evaluation=True)
        outputs = self.ml_cls_task.infer(self.ml_cls_dataset.with_empty_annotations(), inference_parameters)
        for output in outputs:
            assert output.get_annotations()[-1].get_labels()[0].probability == 0.7

    @e2e_pytest_unit
    def test_infer_hierarchicallabel(self, mocker) -> None:
        """Test infer function."""

        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.task.build_dataset",
            return_value=MockDataset(self.hl_cls_dataset),
        )
        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.task.build_dataloader",
            return_value=MockDataLoader(self.hl_cls_dataset),
        )
        mocker.patch.object(MMClassificationTask, "build_model", return_value=MockModel())
        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.task.build_data_parallel",
            return_value=MockModel(),
        )
        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.task.FeatureVectorHook",
            return_value=nullcontext(),
        )

        inference_parameters = InferenceParameters(is_evaluation=True)
        outputs = self.hl_cls_task.infer(self.hl_cls_dataset.with_empty_annotations(), inference_parameters)
        for output in outputs:
            assert output.get_annotations()[-1].get_labels()[0].probability == 0.7

    @e2e_pytest_unit
    def test_cls_evaluate(self) -> None:
        """Test evaluate function for classification."""

        _config = ModelConfiguration(ClassificationConfig("header"), self.mc_cls_label_schema)
        _model = ModelEntity(self.mc_cls_dataset, _config)
        resultset = ResultSetEntity(_model, self.mc_cls_dataset, self.mc_cls_dataset)
        self.mc_cls_task.evaluate(resultset)
        assert resultset.performance.score.value == 1.0

    @e2e_pytest_unit
    def test_cls_evaluate_with_empty_annotations(self) -> None:
        """Test evaluate function for classification with empty predictions."""

        _config = ModelConfiguration(ClassificationConfig("header"), self.mc_cls_label_schema)
        _model = ModelEntity(self.mc_cls_dataset, _config)
        resultset = ResultSetEntity(_model, self.mc_cls_dataset, self.mc_cls_dataset.with_empty_annotations())
        self.mc_cls_task.evaluate(resultset)
        assert resultset.performance.score.value == 0.0

    @pytest.mark.parametrize("precision", [ModelPrecision.FP16, ModelPrecision.FP32])
    @e2e_pytest_unit
    def test_export(self, mocker, precision: ModelPrecision) -> None:
        """Test export function.

        <Steps>
            1. Create model entity
            2. Run export function
            3. Check output model attributes
        """
        _config = ModelConfiguration(ClassificationConfig("header"), self.mc_cls_label_schema)
        _model = ModelEntity(self.mc_cls_dataset, _config)

        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.task.ClassificationExporter",
            return_value=MockExporter(self.mc_cls_task),
        )
        mocker.patch(
            "otx.algorithms.classification.task.embed_ir_model_data",
            return_value=True,
        )

        self.mc_cls_task.export(ExportType.OPENVINO, _model, precision, False)

        assert _model.model_format == ModelFormat.OPENVINO
        assert _model.optimization_type == ModelOptimizationType.MO
        assert _model.precision[0] == precision
        assert _model.get_data("openvino.bin") is not None
        assert _model.get_data("openvino.xml") is not None
        assert _model.precision == self.mc_cls_task._precision
        assert _model.optimization_methods == self.mc_cls_task._optimization_methods
        assert _model.get_data("label_schema.json") is not None

    @e2e_pytest_unit
    def test_explain(self, mocker):
        """Test explain function."""
        explain_parameters = ExplainParameters(
            explainer="ClassWiseSaliencyMap",
            process_saliency_maps=False,
            explain_predicted_classes=True,
        )
        outputs = self.mc_cls_task.explain(self.mc_cls_dataset, explain_parameters)
        assert isinstance(outputs, DatasetEntity)
        assert len(outputs) == 200
