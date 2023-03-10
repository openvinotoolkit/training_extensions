"""Unit Test for otx.algorithms.action.tasks.inference."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
from typing import Any, Dict

import numpy as np
import pytest
import torch
from mmcv.utils import Config
from torch import nn

from otx.algorithms.action.configs.base.configuration import ActionConfig
from otx.algorithms.action.tasks.inference import ActionInferenceTask
from otx.api.configuration import ConfigurableParameters
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
from otx.api.entities.model_template import InstantiationType, TaskFamily, TaskType
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.action.test_helpers import (
    MockModelTemplate,
    generate_action_cls_otx_dataset,
    generate_action_det_otx_dataset,
    generate_labels,
    return_inputs,
)


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

    def evaluate(self, prediction, *args, **kwargs):
        if self.task_type == "cls":
            return {"accuracy": 1.0}
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

    def __init__(self, recipe_cfg, weights, deploy_cfg, work_dir, half_precision):
        self.work_dir = work_dir

    def export(self):
        dummy_data = np.ndarray((1, 1, 1))
        with open(self.work_dir + ".bin", "wb") as f:
            f.write(dummy_data)
        with open(self.work_dir + ".xml", "wb") as f:
            f.write(dummy_data)


class TestActionInferenceTask:
    """Test class for ActionInferenceTask.

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
        cls_template = MockModelTemplate(
            model_template_id="template_id",
            model_template_path="template_path",
            name="template",
            task_family=TaskFamily.VISION,
            task_type=TaskType.ACTION_CLASSIFICATION,
            instantiation=InstantiationType.CLASS,
        )
        self.cls_task_environment = TaskEnvironment(
            model=None,
            hyper_parameters=ConfigurableParameters(header="h-params"),
            label_schema=self.cls_label_schema,
            model_template=cls_template,
        )

        self.cls_dataset = generate_action_cls_otx_dataset(self.video_len, self.frame_len, cls_labels)
        self.cls_task = ActionInferenceTask(task_environment=self.cls_task_environment)

        det_labels = generate_labels(3, Domain.ACTION_DETECTION)
        self.det_label_schema = LabelSchemaEntity()
        det_label_group = LabelGroup(
            name="labels",
            labels=det_labels,
            group_type=LabelGroupType.EXCLUSIVE,
        )
        self.det_label_schema.add_group(det_label_group)
        det_template = MockModelTemplate(
            model_template_id="template_id",
            model_template_path="template_path",
            name="template",
            task_family=TaskFamily.VISION,
            task_type=TaskType.ACTION_DETECTION,
            instantiation=InstantiationType.CLASS,
        )
        self.det_task_environment = TaskEnvironment(
            model=None,
            hyper_parameters=ConfigurableParameters(header="h-params"),
            label_schema=self.det_label_schema,
            model_template=det_template,
        )

        self.det_dataset = generate_action_det_otx_dataset(self.video_len, self.frame_len, det_labels)[0]
        self.det_task = ActionInferenceTask(task_environment=self.det_task_environment)

    @e2e_pytest_unit
    # TODO Sepearate add prediction function test and infer funciton test
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
            "otx.algorithms.action.tasks.inference.build_dataset", return_value=MockDataset(self.cls_dataset, "cls")
        )
        mocker.patch(
            "otx.algorithms.action.tasks.inference.build_dataloader", return_value=MockDataLoader(self.cls_dataset)
        )
        mocker.patch("otx.algorithms.action.tasks.inference.MMDataParallel", return_value=MockModel("cls"))
        self.cls_task._model = MockModel("cls")
        self.cls_task._recipe_cfg = Config(
            {
                "data": {"test": {"otx_dataset": None}, "workers_per_gpu": 1},
                "gpu_ids": [0],
                "evaluation": {"final_metric": "accuracy"},
            }
        )
        inference_parameters = InferenceParameters(is_evaluation=True)
        outputs = self.cls_task.infer(self.cls_dataset, inference_parameters)
        for output in outputs:
            assert len(output.get_annotations()[0].get_labels()) == 2

        mocker.patch(
            "otx.algorithms.action.tasks.inference.build_dataset", return_value=MockDataset(self.det_dataset, "det")
        )
        mocker.patch(
            "otx.algorithms.action.tasks.inference.build_dataloader", return_value=MockDataLoader(self.det_dataset)
        )
        mocker.patch("otx.algorithms.action.tasks.inference.MMDataParallel", return_value=MockModel("det"))
        self.det_task._model = MockModel("det")
        self.det_task._recipe_cfg = Config(
            {
                "data": {"test": {"otx_dataset": None}, "workers_per_gpu": 1},
                "gpu_ids": [0],
                "evaluation": {"final_metric": "mAP@0.5IOU"},
            }
        )
        inference_parameters = InferenceParameters(is_evaluation=True)
        outputs = self.det_task.infer(self.det_dataset, inference_parameters)
        for output in outputs:
            assert len(output.get_annotations()) == 4

    @e2e_pytest_unit
    def test_evaluate(self) -> None:
        """Test evaluate function.

        <Steps>
            1. Create model entity
            2. Create result set entity
            3. Run evaluate function with same dataset, this should give 100% accuracy
            4. Run evaluate function with empty dataset, this should give 0% accuracy
            5. Do 1 - 4 for action detection
        """
        _config = ModelConfiguration(ActionConfig(), self.cls_label_schema)
        _model = ModelEntity(self.cls_dataset, _config)
        resultset = ResultSetEntity(_model, self.cls_dataset, self.cls_dataset)
        self.cls_task.evaluate(resultset)
        assert resultset.performance.score.value == 1.0

        resultset = ResultSetEntity(_model, self.cls_dataset, self.cls_dataset.with_empty_annotations())
        self.cls_task.evaluate(resultset)
        assert resultset.performance.score.value == 0.0

        _config = ModelConfiguration(ActionConfig(), self.det_label_schema)
        _model = ModelEntity(self.det_dataset, _config)
        resultset = ResultSetEntity(_model, self.det_dataset, self.det_dataset)
        self.det_task.evaluate(resultset)
        assert resultset.performance.score.value == 0.0

    @e2e_pytest_unit
    def test_initialize_post_hook(self) -> None:
        """Test _initialize_post_hook funciton."""

        options = None
        assert self.cls_task._initialize_post_hook(options) is None

        options = {"deploy_cfg": Config()}
        self.cls_task._initialize_post_hook(options)
        assert isinstance(self.cls_task.deploy_cfg, Config)

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
        self.cls_task._model = nn.Module()
        self.cls_task._recipe_cfg = None
        self.cls_task.deploy_cfg = Config(
            dict(codebase_config=dict(type="mmdet", task="ObjectDetection"), backend_config=dict(type="openvino"))
        )
        mocker.patch("otx.algorithms.action.tasks.inference.ActionInferenceTask._init_task", return_value=True)
        mocker.patch("otx.algorithms.action.tasks.inference.Exporter", side_effect=MockExporter)
        self.cls_task.export(ExportType.OPENVINO, _model, precision)

        assert _model.model_format == ModelFormat.OPENVINO
        assert _model.optimization_type == ModelOptimizationType.MO
        assert _model.precision[0] == precision
        assert _model.get_data("openvino.bin") is not None
        assert _model.get_data("openvino.xml") is not None
        assert _model.get_data("confidence_threshold") is not None
        assert _model.precision == self.cls_task._precision
        assert _model.optimization_methods == self.cls_task._optimization_methods
        assert _model.get_data("label_schema.json") is not None

    @e2e_pytest_unit
    def test_init_task(self, mocker) -> None:
        """Test _init_task function.

        Check model is generated from _init_task function.
        """
        mocker.patch("otx.algorithms.action.tasks.inference.ActionInferenceTask._initialize", return_value=True)
        mocker.patch(
            "otx.algorithms.action.tasks.inference.ActionInferenceTask._load_model", return_value=MockModel("cls")
        )
        with pytest.raises(RuntimeError):
            self.cls_task._init_task()

        self.cls_task._recipe_cfg = Config()
        self.cls_task._init_task()
        assert isinstance(self.cls_task._model, MockModel)

    @e2e_pytest_unit
    def test_load_model(self, mocker) -> None:
        """Test _load_model function.

        Check _load_model function can run _create_model function under various situations
        """

        mocker.patch(
            "otx.algorithms.action.tasks.inference.ActionInferenceTask._create_model", return_value=MockModel("cls")
        )
        mocker.patch("otx.algorithms.action.tasks.inference.load_state_dict", return_value=True)
        mocker.patch(
            "otx.algorithms.action.tasks.inference.torch.load",
            return_value={"confidence_threshold": 0.01, "model": np.array([0.01])},
        )

        with pytest.raises(Exception):
            self.cls_task._load_model(None)

        _config = ModelConfiguration(ActionConfig(), self.cls_label_schema)
        _model_entity = ModelEntity(self.cls_dataset, _config)
        _model_entity.set_data("weights.pth", np.ndarray((1, 1, 1)).tobytes())

        self.cls_task._recipe_cfg = Config({"load_from": "weights.pth"})
        model = self.cls_task._load_model(_model_entity)
        assert isinstance(model, MockModel)

        model = self.cls_task._load_model(None)
        assert isinstance(model, MockModel)

    def test_create_model(self, mocker) -> None:
        """Test _create_model function.

        Check _create_model function can run build_model funciton under various situations
        """
        mocker.patch("otx.algorithms.action.tasks.inference.build_model", return_value=MockModel("cls"))
        mocker.patch("otx.algorithms.action.tasks.inference.load_checkpoint", return_value=True)

        _config = Config({"model": Config(), "load_from": "weights.pth"})
        model = self.cls_task._create_model(_config, False)
        assert isinstance(model, MockModel)
        model = self.cls_task._create_model(_config, True)
        assert isinstance(model, MockModel)

    def test_unload(self) -> None:
        """Test unload function."""
        self.cls_task.unload()

    def test_init_recipe_hparam(self, mocker) -> None:
        """Test _init_recipe_hparam function."""

        mocker.patch(
            "otx.algorithms.action.tasks.inference.BaseTask._init_recipe_hparam",
            return_value=Config(
                {"data": {"samples_per_gpu": 4}, "lr_config": {"warmup_iters": 3}, "runner": {"max_epochs": 10}}
            ),
        )

        self.cls_task._recipe_cfg = Config({"lr_config": Config()})
        out = self.cls_task._init_recipe_hparam()

        assert self.cls_task._recipe_cfg.lr_config.warmup == "linear"
        assert self.cls_task._recipe_cfg.lr_config.warmup_by_epoch is True
        assert self.cls_task._recipe_cfg.total_epochs == 10
        assert out.data.videos_per_gpu == 4
        assert out.use_adaptive_interval == self.cls_task._hyperparams.learning_parameters.use_adaptive_interval

    def test_init_recipe(self, mocker) -> None:
        """Test _init_recipe funciton."""

        mocker.patch("otx.algorithms.action.tasks.inference.Config.fromfile", side_effect=return_inputs)
        mocker.patch("otx.algorithms.action.tasks.inference.patch_config", return_value=True)
        mocker.patch("otx.algorithms.action.tasks.inference.set_data_classes", return_value=True)

        self.cls_task._init_recipe()
        recipe_root = os.path.abspath(os.path.dirname(self.cls_task.template_file_path))
        assert self.cls_task._recipe_cfg == os.path.join(recipe_root, "model.py")

    def test_init_model_cfg(self, mocker) -> None:
        """Test _init_model_cfg function."""

        mocker.patch("otx.algorithms.action.tasks.inference.Config.fromfile", side_effect=return_inputs)

        model_cfg = self.cls_task._init_model_cfg()
        assert model_cfg == os.path.join(self.cls_task._model_dir, "model.py")
