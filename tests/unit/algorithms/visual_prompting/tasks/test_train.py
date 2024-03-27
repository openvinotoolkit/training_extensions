"""Tests the methods in the train task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest

from otx.algorithms.visual_prompting.tasks.train import TrainingTask
from otx.api.entities.metrics import NullPerformance
from otx.api.entities.model import ModelEntity
from otx.api.entities.train_parameters import TrainParameters
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.visual_prompting.test_helpers import (
    generate_visual_prompting_dataset,
    init_environment,
)


class TestTrainingTask:
    @pytest.fixture(autouse=True)
    def setup(self, tmpdir, mocker):
        mocker.patch("pathlib.Path.write_text")
        self.task_environment = init_environment()
        self.output_path = str(tmpdir.mkdir("visual_prompting_training_test"))
        self.training_task = TrainingTask(self.task_environment, self.output_path)

    @e2e_pytest_unit
    def test_train(self, mocker):
        """Test train."""
        mocker_trainer = mocker.patch("otx.algorithms.visual_prompting.tasks.train.Trainer")
        mocker_save = mocker.patch("torch.save")
        mocker.patch.object(self.training_task, "model_info")

        dataset = generate_visual_prompting_dataset()
        output_model = ModelEntity(
            dataset,
            self.task_environment.get_model_configuration(),
        )

        self.training_task.train(dataset, output_model, TrainParameters())

        mocker_trainer.assert_called_once()
        mocker_save.assert_called_once()
        assert not isinstance(output_model.performance, NullPerformance)
        assert output_model.model_adapters.get("weights.pth", None)
        assert output_model.model_adapters.get("label_schema.json", None)
