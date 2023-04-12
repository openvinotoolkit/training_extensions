"""Unit Test for otx.algorithms.action.adapters.mmaction.task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
from copy import deepcopy
from typing import Any, Dict

import numpy as np
import pytest
import torch
from mmaction.models.backbones.x3d import X3D
from mmaction.models.recognizers.recognizer3d import Recognizer3D
from mmcv.runner import CheckpointLoader
from mmcv.utils import Config
from torch import nn

from otx.algorithms.action.configs.base.configuration import ActionConfig
from otx.algorithms.action.adapters.mmaction.task import MMActionTask
from otx.algorithms.common.adapters.mmcv.utils.config_utils import MPAConfig
from otx.api.configuration import ConfigurableParameters
from otx.api.configuration.helper import create
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
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
from tests.unit.algorithms.action.test_helpers import (
    init_environment,
    MockModelTemplate,
    generate_action_cls_otx_dataset,
    generate_action_det_otx_dataset,
    generate_labels,
    return_inputs,
)

DEFAULT_ACTION_CLS_DIR = os.path.join("otx/algorithms/action/configs/classification", "x3d")
DEFAULT_ACTION_DET_DIR = os.path.join("otx/algorithms/action/configs/detection", "x3d_fast_rcnn")


class MockModule(nn.Module):
    """Mock class for nn.Module."""

    def forward(self, inputs: Any):
        return inputs


class MockModel(nn.Module):
    """Mock class for pytorch model."""

    def __init__(self, task_type):
        super().__init__()
        self.module = MockModule()
        self.module.backbone = MockModule()
        self.backbone = MockModule()
        self.task_type = task_type

    def forward(self, return_loss: bool, imgs: DatasetItemEntity):
        forward_hooks = list(self.module.backbone._forward_hooks.values())
        for hook in forward_hooks:
            hook(1, 2, 3)
        if self.task_type == "cls":
            return np.array([[0, 0, 1]])
        return [[np.array([[0, 0, 1, 1, 0.1]]), np.array([[0, 0, 1, 1, 0.2]]), np.array([[0, 0, 1, 1, 0.7]])]]

    @staticmethod
    def named_parameters():
        return {"name": torch.Tensor([0.5])}.items()


class MockDataset(DatasetEntity):
    """Mock class for mm_dataset."""

    def __init__(self, dataset: DatasetEntity, task_type: str):
        self.dataset = dataset
        self.task_type = task_type
        self.CLASSES = ["1", "2", "3"]

    def evaluate(self, prediction, *args, **kwargs):
        if self.task_type == "cls":
            return {"mean_class_accuracy": 1.0}
        else:
            return {"mAP@0.5IOU": 1.0}


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
        self.work_dir = task._output_path

    def export(self):
        dummy_data = np.ndarray((1, 1, 1))
        with open(os.path.join(self.work_dir, "openvino.bin"), "wb") as f:
            f.write(dummy_data)
        with open(os.path.join(self.work_dir, "openvino.xml"), "wb") as f:
            f.write(dummy_data)
        with open(os.path.join(self.work_dir, "model.onnx"), "wb") as f:
            f.write(dummy_data)


class TestMMActionTask:
    """Test class for MMActionTask.

    Details are explained in each test function.
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.video_len = 3
        self.frame_len = 3

        cls_labels = generate_labels(3, Domain.ACTION_CLASSIFICATION)
        self.cls_label_schema = LabelSchemaEntity()
        cls_label_group = LabelGroup(
            name="labels",
            labels=cls_labels,
            group_type=LabelGroupType.EXCLUSIVE,
        )
        self.cls_label_schema.add_group(cls_label_group)
        self.cls_dataset = generate_action_cls_otx_dataset(self.video_len, self.frame_len, cls_labels)

        cls_model_template = parse_model_template(os.path.join(DEFAULT_ACTION_CLS_DIR, "template.yaml"))
        cls_hyper_parameters = create(cls_model_template.hyper_parameters.data)
        cls_task_env = init_environment(cls_hyper_parameters, cls_model_template, self.cls_label_schema)
        self.cls_task = MMActionTask(task_environment=cls_task_env)

        det_labels = generate_labels(3, Domain.ACTION_DETECTION)
        self.det_label_schema = LabelSchemaEntity()
        det_label_group = LabelGroup(
            name="labels",
            labels=det_labels,
            group_type=LabelGroupType.EXCLUSIVE,
        )
        self.det_label_schema.add_group(det_label_group)
        self.det_dataset = generate_action_det_otx_dataset(self.video_len, self.frame_len, det_labels)[0]

        det_model_template = parse_model_template(os.path.join(DEFAULT_ACTION_DET_DIR, "template.yaml"))
        det_hyper_parameters = create(det_model_template.hyper_parameters.data)
        det_task_env = init_environment(det_hyper_parameters, det_model_template, self.det_label_schema)
        self.det_task = MMActionTask(task_environment=det_task_env)

    @e2e_pytest_unit
    def test_build_model(self, mocker) -> None:
        """Test build_model function."""
        _weight = torch.randn([24, 3, 1, 3, 3])
        mocker.patch.object(
            CheckpointLoader,
            "load_checkpoint",
            return_value={"model": {"state_dict": {"backbone.conv1_s.conv.weight": _weight}}},
        )
        _mock_recipe_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_ACTION_CLS_DIR, "model.py"))
        model = self.cls_task.build_model(_mock_recipe_cfg, True)
        assert isinstance(model, Recognizer3D)
        assert isinstance(model.backbone, X3D)
        assert torch.all(model.state_dict()["backbone.conv1_s.conv.weight"] == _weight)

    @e2e_pytest_unit
    def test_train(self, mocker) -> None:
        """Test train function."""
        mocker.patch(
            "otx.algorithms.action.adapters.mmaction.task.build_dataset",
            return_value=MockDataset(self.cls_dataset, "cls"),
        )
        mocker.patch(
            "otx.algorithms.action.adapters.mmaction.task.build_dataloader",
            return_value=MockDataLoader(self.cls_dataset),
        )
        mocker.patch.object(MMActionTask, "build_model", return_value=MockModel("cls"))
        mocker.patch(
            "otx.algorithms.action.adapters.mmaction.task.patch_data_pipeline",
            return_value=True,
        )
        mocker.patch(
            "otx.algorithms.action.adapters.mmaction.task.build_data_parallel",
            return_value=MockModel("cls"),
        )
        mocker.patch(
            "otx.algorithms.action.adapters.mmaction.task.train_model",
            return_value=True,
        )
        mocker.patch("torch.load", return_value={"state_dict": np.ndarray([1, 1, 1])})

        _config = ModelConfiguration(ActionConfig(), self.cls_label_schema)
        output_model = ModelEntity(self.cls_dataset, _config)
        self.cls_task.train(self.cls_dataset, output_model)
        output_model.performance == 1.0

        mocker.patch(
            "otx.algorithms.action.adapters.mmaction.task.build_dataset",
            return_value=MockDataset(self.det_dataset, "det"),
        )
        mocker.patch(
            "otx.algorithms.action.adapters.mmaction.task.build_dataloader",
            return_value=MockDataLoader(self.det_dataset),
        )
        mocker.patch.object(MMActionTask, "build_model", return_value=MockModel("det"))
        mocker.patch(
            "otx.algorithms.action.adapters.mmaction.task.patch_data_pipeline",
            return_value=True,
        )
        mocker.patch(
            "otx.algorithms.action.adapters.mmaction.task.build_data_parallel",
            return_value=MockModel("det"),
        )
        _config = ModelConfiguration(ActionConfig(), self.det_label_schema)
        output_model = ModelEntity(self.det_dataset, _config)
        self.det_task.train(self.det_dataset, output_model)
        output_model.performance == 1.0

    @e2e_pytest_unit
    def test_infer(self, mocker) -> None:
        """Test infer function.

        <Steps>
            1. Create mock model for action classification
            2. Create mock recipe for action classification
            3. Run infer funciton
            4. Check whether inference results are added to output
            5. Do 1 - 4 for action detection
        """
        mocker.patch(
            "otx.algorithms.action.adapters.mmaction.task.build_dataset",
            return_value=MockDataset(self.cls_dataset, "cls"),
        )
        mocker.patch(
            "otx.algorithms.action.adapters.mmaction.task.build_dataloader",
            return_value=MockDataLoader(self.cls_dataset),
        )
        mocker.patch.object(MMActionTask, "build_model", return_value=MockModel("cls"))
        mocker.patch(
            "otx.algorithms.action.adapters.mmaction.task.patch_data_pipeline",
            return_value=True,
        )
        mocker.patch(
            "otx.algorithms.action.adapters.mmaction.task.build_data_parallel",
            return_value=MockModel("cls"),
        )

        inference_parameters = InferenceParameters(is_evaluation=True)
        outputs = self.cls_task.infer(self.cls_dataset, inference_parameters)
        for output in outputs:
            assert len(output.get_annotations()[0].get_labels()) == 2

        mocker.patch(
            "otx.algorithms.action.adapters.mmaction.task.build_dataset",
            return_value=MockDataset(self.det_dataset, "det"),
        )
        mocker.patch(
            "otx.algorithms.action.adapters.mmaction.task.build_dataloader",
            return_value=MockDataLoader(self.det_dataset),
        )
        mocker.patch.object(MMActionTask, "build_model", return_value=MockModel("det"))
        mocker.patch(
            "otx.algorithms.action.adapters.mmaction.task.patch_data_pipeline",
            return_value=True,
        )
        mocker.patch(
            "otx.algorithms.action.adapters.mmaction.task.build_data_parallel",
            return_value=MockModel("det"),
        )
        inference_parameters = InferenceParameters(is_evaluation=True)
        outputs = self.det_task.infer(self.det_dataset, inference_parameters)
        for output in outputs:
            assert len(output.get_annotations()) == 2

    @e2e_pytest_unit
    def test_evaluate(self) -> None:
        """Test evaluate function."""
        _config = ModelConfiguration(ActionConfig(), self.cls_label_schema)
        _model = ModelEntity(self.cls_dataset, _config)
        resultset = ResultSetEntity(_model, self.cls_dataset, self.cls_dataset)
        self.cls_task.evaluate(resultset)
        assert resultset.performance.score.value == 1.0

    @e2e_pytest_unit
    def test_evaluate_with_empty_annot(self) -> None:
        """Test evaluate function with empty_annot."""
        _config = ModelConfiguration(ActionConfig(), self.cls_label_schema)
        _model = ModelEntity(self.cls_dataset, _config)
        resultset = ResultSetEntity(_model, self.cls_dataset, self.cls_dataset.with_empty_annotations())
        self.cls_task.evaluate(resultset)
        assert resultset.performance.score.value == 0.0

    @e2e_pytest_unit
    def test_evaluate_det(self) -> None:
        """Test evaluate function for action detection."""
        _config = ModelConfiguration(ActionConfig(), self.det_label_schema)
        _model = ModelEntity(self.det_dataset, _config)
        resultset = ResultSetEntity(_model, self.det_dataset, self.det_dataset)
        self.det_task.evaluate(resultset)
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
        _config = ModelConfiguration(ActionConfig(), self.cls_label_schema)
        _model = ModelEntity(self.cls_dataset, _config)
        mocker.patch("otx.algorithms.action.adapters.mmaction.task.Exporter", return_value=MockExporter(self.cls_task))
        mocker.patch("torch.load", return_value={})
        mocker.patch("torch.nn.Module.load_state_dict", return_value=True)

        self.cls_task.export(ExportType.OPENVINO, _model, precision, False)

        assert _model.model_format == ModelFormat.OPENVINO
        assert _model.optimization_type == ModelOptimizationType.MO
        assert _model.precision[0] == precision
        assert _model.get_data("openvino.bin") is not None
        assert _model.get_data("openvino.xml") is not None
        assert _model.get_data("confidence_threshold") is not None
        assert _model.precision == self.cls_task._precision
        assert _model.optimization_methods == self.cls_task._optimization_methods
        assert _model.get_data("label_schema.json") is not None
