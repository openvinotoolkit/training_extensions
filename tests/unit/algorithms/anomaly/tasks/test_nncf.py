"""Tests the methods in the NNCF task."""

# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from otx.algorithms.anomaly.tasks.nncf import NNCFTask
from otx.algorithms.anomaly.tasks.train import TrainingTask
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.model import ModelEntity, ModelOptimizationType
from otx.api.entities.model_template import TaskType
from otx.api.entities.train_parameters import TrainParameters
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from otx.api.usecases.tasks.interfaces.optimization_interface import OptimizationType
from tests.unit.algorithms.anomaly.helpers.dummy_dataset import get_shapes_dataset
from tests.unit.algorithms.anomaly.helpers.utils import create_task_environment


class TestNNCFTask:
    """Tests methods in the NNCF task."""

    @pytest.mark.parametrize(
        "task_type", [TaskType.ANOMALY_CLASSIFICATION, TaskType.ANOMALY_DETECTION, TaskType.ANOMALY_SEGMENTATION]
    )
    def test_nncf(self, task_type, tmpdir):
        """Tests the NNCF optimize method."""
        root = str(tmpdir.mkdir("anomaly_nncf_test"))

        dataset: DatasetEntity = get_shapes_dataset(task_type, one_each=True)

        task_environment = create_task_environment(dataset, task_type)

        output_model = ModelEntity(
            dataset,
            task_environment.get_model_configuration(),
        )

        train_task = TrainingTask(task_environment, output_path=root)
        train_task.train(dataset, output_model, TrainParameters())
        task_environment.model = output_model

        # create normal nncf task
        nncf_task = NNCFTask(task_environment, output_path=root)

        # optimize the model
        assert output_model.optimization_type == ModelOptimizationType.NONE
        optimized_model = ModelEntity(dataset, configuration=task_environment.get_model_configuration())
        nncf_task.optimize(OptimizationType.NNCF, dataset, optimized_model)
        assert optimized_model.optimization_type == ModelOptimizationType.NNCF

        # load the optimized model
        new_nncf_task = NNCFTask(task_environment, output_path=root)
        assert new_nncf_task.compression_ctrl is None
        new_nncf_task.model = new_nncf_task.load_model(optimized_model)
        assert new_nncf_task.compression_ctrl is not None

        # Export model
        new_nncf_task.export(ExportType.OPENVINO, optimized_model)
        optimized_model.get_data("openvino.bin")  # Should not raise an exception
