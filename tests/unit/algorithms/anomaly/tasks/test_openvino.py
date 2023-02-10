"""Tests the methods in the OpenVINO task."""

# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from otx.algorithms.anomaly.tasks.openvino import OpenVINOTask
from otx.algorithms.anomaly.tasks.train import TrainingTask
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.model import ModelEntity, ModelOptimizationType
from otx.api.entities.model_template import TaskType
from otx.api.entities.optimization_parameters import OptimizationParameters
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.subset import Subset
from otx.api.entities.train_parameters import TrainParameters
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from otx.api.usecases.tasks.interfaces.optimization_interface import OptimizationType
from tests.unit.algorithms.anomaly.helpers.dummy_dataset import get_shapes_dataset
from tests.unit.algorithms.anomaly.helpers.utils import create_task_environment


class TestOpenVINOTask:
    """Tests methods in the OpenVINO task."""

    @pytest.mark.parametrize(
        "task_type", [TaskType.ANOMALY_CLASSIFICATION, TaskType.ANOMALY_DETECTION, TaskType.ANOMALY_SEGMENTATION]
    )
    def test_openvino(self, task_type, tmpdir):
        """Tests the OpenVINO optimize method."""
        root = str(tmpdir.mkdir("anomaly_openvino_test"))

        dataset: DatasetEntity = get_shapes_dataset(task_type, one_each=True)
        dataset = dataset.get_subset(Subset.VALIDATION)

        task_environment = create_task_environment(dataset, task_type)

        output_model = ModelEntity(
            dataset,
            task_environment.get_model_configuration(),
        )

        train_task = TrainingTask(task_environment, output_path=root)
        train_task.train(dataset, output_model, TrainParameters())
        task_environment.model = output_model

        train_task.export(ExportType.OPENVINO, output_model)

        # Create OpenVINO task
        openvino_task = OpenVINOTask(task_environment)

        # call inference
        predicted_dataset = openvino_task.infer(
            dataset.with_empty_annotations(), InferenceParameters(is_evaluation=True)
        )

        # call evaluate
        result_set = ResultSetEntity(output_model, dataset, predicted_dataset)
        openvino_task.evaluate(result_set)
        if task_type in (TaskType.ANOMALY_CLASSIFICATION, TaskType.ANOMALY_DETECTION):
            assert result_set.performance.score.name == "f-measure"
        elif task_type == TaskType.ANOMALY_SEGMENTATION:
            assert result_set.performance.score.name == "Dice Average"

        # optimize to POT
        openvino_task.optimize(OptimizationType.POT, dataset, output_model, OptimizationParameters())
        assert output_model.optimization_type == ModelOptimizationType.POT
        assert output_model.get_data("label_schema.json") is not None

        # deploy
        openvino_task.deploy(output_model)
        assert output_model.exportable_code is not None
