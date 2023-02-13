"""Unit Test for otx.algorithms.action.tasks.train."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os

import pytest
import torch
from mmcv.utils import Config

from otx.algorithms.action.configs.base.configuration import ActionConfig
from otx.algorithms.action.tasks.train import ActionTrainTask
from otx.api.configuration import ConfigurableParameters
from otx.api.entities.label import Domain
from otx.api.entities.label_schema import LabelGroup, LabelGroupType, LabelSchemaEntity
from otx.api.entities.metrics import (
    BarMetricsGroup,
    LineMetricsGroup,
    Performance,
    ScoreMetric,
)
from otx.api.entities.model import ModelConfiguration, ModelEntity
from otx.api.entities.model_template import InstantiationType, TaskFamily, TaskType
from otx.api.entities.task_environment import TaskEnvironment
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.action.test_helpers import (
    MockModelTemplate,
    generate_action_cls_otx_dataset,
    generate_labels,
)


class TestActionTrainTask:
    """Test class for ActionTrainTask class."""

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
        self.cls_task = ActionTrainTask(task_environment=self.cls_task_environment)

    @e2e_pytest_unit
    def test_save_model(self, mocker) -> None:
        """Test save_model function."""

        mocker.patch("otx.algorithms.action.tasks.train.torch.load", return_value={"state_dict": None})

        config = ModelConfiguration(ActionConfig(), self.cls_label_schema)
        output_model = ModelEntity(self.cls_dataset, config)
        self.cls_task.save_model(output_model)
        assert output_model.get_data("weights.pth") is not None
        assert output_model.get_data("label_schema.json") is not None
        assert output_model.precision == self.cls_task._precision

    @e2e_pytest_unit
    def test_cancel_training(self) -> None:
        """Test cance_trainng function."""

        class _MockCanceInterface:
            def cancel(self):
                raise RuntimeError("Checking for calling this function")

        self.cls_task.cancel_training()
        assert self.cls_task.reserved_cancel is True

        self.cls_task.cancel_interface = _MockCanceInterface()
        with pytest.raises(RuntimeError):
            self.cls_task.cancel_training()

    @e2e_pytest_unit
    def test_train(self, mocker) -> None:
        """Test train function."""

        config = ModelConfiguration(ActionConfig(), self.cls_label_schema)
        output_model = ModelEntity(self.cls_dataset, config)
        self.cls_task._should_stop = True
        self.cls_task.train(self.cls_dataset, output_model)
        assert self.cls_task._should_stop is False
        assert self.cls_task._is_training is False

        mocker.patch("otx.algorithms.action.tasks.train.ActionTrainTask._init_task", return_value=True)
        mocker.patch("otx.algorithms.action.tasks.train.ActionTrainTask._train_model", return_value=True)
        mocker.patch("otx.algorithms.action.tasks.train.ActionTrainTask._get_output_model", return_value=True)
        mocker.patch("otx.algorithms.action.tasks.train.ActionTrainTask._get_final_eval_results", return_value=0.5)
        mocker.patch("otx.algorithms.action.tasks.train.ActionTrainTask.save_model", return_value=True)

        self.cls_task._recipe_cfg = None
        self.cls_task._should_stop = False
        with pytest.raises(Exception):
            self.cls_task.train(self.cls_dataset, output_model)
        self.cls_task._recipe_cfg = Config()
        self.cls_task.train(self.cls_dataset, output_model)
        assert output_model.performance == 0.5
        assert self.cls_task._is_training is False

    @e2e_pytest_unit
    def test_train_model(self, mocker) -> None:
        """Test _train_model function."""

        with pytest.raises(Exception):
            self.cls_task._train_model(self.cls_dataset)

        self.cls_task._recipe_cfg = Config({"work_dir": self.cls_task._output_path})
        self.cls_task._model = torch.nn.Module()

        def _mock_train_model(*args, **kwargs):
            with open(os.path.join(self.cls_task._recipe_cfg.work_dir, "best.pth"), "wb") as f:
                torch.save(torch.randn(1), f)

        mocker.patch(
            "otx.algorithms.action.tasks.train.prepare_for_training", return_value=Config({"data": {"train": None}})
        )
        mocker.patch("otx.algorithms.action.tasks.train.build_dataset", return_value=True)
        mocker.patch("otx.algorithms.action.tasks.train.train_model", side_effect=_mock_train_model)

        out = self.cls_task._train_model(self.cls_dataset)
        assert out["final_ckpt"] is not None
        assert out["final_ckpt"].split("/")[-1] == "best.pth"

    @e2e_pytest_unit
    def test_get_output_model(self, mocker) -> None:
        """Test _get_output_model function."""

        self.cls_task._model = torch.nn.Module()
        sample_results = {"final_ckpt": None}
        self.cls_task._get_output_model(sample_results)

        mocker.patch("otx.algorithms.action.tasks.train.torch.load", return_value={"state_dict": {}})
        sample_results = {"final_ckpt": "checkpoint_file_path"}
        self.cls_task._get_output_model(sample_results)

    @e2e_pytest_unit
    def test_get_final_eval_results(self, mocker) -> None:
        """Test _get_final_eval_results."""

        class _mock_metric:
            def __init__(self):
                self.performance = Performance(ScoreMetric("accuracy", 1.0))

            def get_performance(self):
                return self.performance

        config = ModelConfiguration(ActionConfig(), self.cls_label_schema)
        output_model = ModelEntity(self.cls_dataset, config)

        mocker.patch("otx.algorithms.action.tasks.train.ActionTrainTask._infer_model", return_value=(True, True))
        mocker.patch("otx.algorithms.action.tasks.train.ActionTrainTask._add_predictions_to_dataset", return_value=True)
        mocker.patch(
            "otx.algorithms.action.tasks.train.ActionTrainTask._add_det_predictions_to_dataset", return_value=True
        )
        mocker.patch("otx.algorithms.action.tasks.train.ActionTrainTask._generate_training_metrics", return_value=[])

        mocker.patch("otx.algorithms.action.tasks.train.ActionTrainTask._get_metric", return_value=_mock_metric())
        self.cls_task._recipe_cfg = Config({"evaluation": {"final_metric": "accuracy"}})
        performance = self.cls_task._get_final_eval_results(self.cls_dataset, output_model)
        assert performance.score.name == "accuracy"
        assert performance.score.value == 1.0

        self.cls_task._task_type = TaskType.ACTION_DETECTION
        performance = self.cls_task._get_final_eval_results(self.cls_dataset, output_model)

    @e2e_pytest_unit
    def test_generate_training_metrics(self) -> None:
        """Test _generate_training_metrics fucntion."""

        class MockCurve:
            def __init__(self, x: list, y: list):
                self.x = x
                self.y = y

        sample_learning_curve = {
            "dummy0": MockCurve([0, 1, 2], [2, 1, 0]),
            "dummy1": MockCurve([0, 1, 2], [2, 1]),
        }
        output = self.cls_task._generate_training_metrics(sample_learning_curve, 1.0, "accuracy")
        assert isinstance(output[0], LineMetricsGroup)
        assert isinstance(output[-1], BarMetricsGroup)
