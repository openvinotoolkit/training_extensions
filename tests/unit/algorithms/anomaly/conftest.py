"""Fixtures for anomaly tests."""
# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest

from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.model import ModelEntity
from otx.api.entities.model_template import TaskType
from otx.api.entities.task_environment import TaskEnvironment
from tests.unit.algorithms.anomaly.helpers.dummy_dataset import get_shapes_dataset
from tests.unit.algorithms.anomaly.helpers.utils import create_task_environment


@dataclass(frozen=True)  # this ensures that the objects are immutable across tests
class TestEnvironment:
    """Test environment for anomaly tests."""

    task_environment: TaskEnvironment
    output_model: ModelEntity
    dataset: DatasetEntity
    task_type: TaskType


@pytest.fixture(
    scope="session", params=[TaskType.ANOMALY_CLASSIFICATION, TaskType.ANOMALY_DETECTION, TaskType.ANOMALY_SEGMENTATION]
)
def setup_task_environment(request):
    """Returns a task environment, a model and datset."""
    task_type = request.param
    dataset: DatasetEntity = get_shapes_dataset(task_type, one_each=True)
    task_environment = create_task_environment(dataset, task_type)
    output_model = ModelEntity(
        dataset,
        task_environment.get_model_configuration(),
    )
    environment = TestEnvironment(task_environment, output_model, dataset, task_type)
    return environment
