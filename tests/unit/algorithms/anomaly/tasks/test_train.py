"""Tests the methods in the train task."""

# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from copy import deepcopy

from torch import nn

from otx.algorithms.anomaly.tasks.train import TrainingTask
from otx.api.entities.train_parameters import TrainParameters
from tests.unit.algorithms.anomaly.helpers.utils import create_task_environment


class TestTrainTask:
    """Tests the methods in the train task."""

    def _compare_state_dict(self, model1: nn.Module, model2: nn.Module):
        """Compares the state dict of two models."""
        state_dict1 = model1.state_dict()
        state_dict2 = model2.state_dict()
        for (key1, param1), (key2, param2) in zip(state_dict1.items(), state_dict2.items()):
            assert key1 == key2
            if not param1.data.isnan().any() and "bn" not in key1:
                assert param1.data.allclose(param2.data)

    @pytest.mark.skip(reason="CVS-107918 FAIL code -11 in anomaly unit test on python3.10")
    def test_train_and_load(self, tmpdir, setup_task_environment):
        """Tests the train method and check if it can be loaded correctly."""
        root = str(tmpdir.mkdir("anomaly_training_test"))

        # Get task environment
        setup_task_environment = deepcopy(setup_task_environment)  # since fixture is mutable
        task_environment = setup_task_environment.task_environment
        output_model = setup_task_environment.output_model
        dataset = setup_task_environment.dataset
        task_type = setup_task_environment.task_type

        train_task = TrainingTask(task_environment, output_path=root)
        train_task.train(dataset, output_model, TrainParameters())
        train_task.model = train_task.load_model(output_model)

        # create a new output_model and load from task environment
        # check if the loaded model is the same as the trained model
        new_task_environment = create_task_environment(dataset, task_type)
        new_task_environment.model = output_model
        new_task = TrainingTask(new_task_environment, output_path=root)  # should load the model from the output_model
        self._compare_state_dict(train_task.model, new_task.model)
