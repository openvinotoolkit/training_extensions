"""Tests the methods in the OpenVINO task."""

# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import numpy as np

from otx.algorithms.anomaly.tasks.openvino import OpenVINOTask
from otx.algorithms.anomaly.tasks.train import TrainingTask
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.model import ModelEntity, ModelOptimizationType
from otx.api.entities.model_template import TaskType
from otx.api.entities.optimization_parameters import OptimizationParameters
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.subset import Subset
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from otx.api.usecases.tasks.interfaces.optimization_interface import OptimizationType


class TestOpenVINOTask:
    """Tests methods in the OpenVINO task."""

    def set_normalization_params(self, output_model: ModelEntity):
        """Sets normalization parameters for an untrained output model.

        This is needed as untrained model might have nan values for normalization parameters which will raise an error.
        """
        output_model.set_data("image_threshold", np.float32(0.5).tobytes())
        output_model.set_data("pixel_threshold", np.float32(0.5).tobytes())
        output_model.set_data("min", np.float32(0).tobytes())
        output_model.set_data("max", np.float32(1).tobytes())

    def test_openvino(self, tmpdir, setup_task_environment):
        """Tests the OpenVINO optimize method."""
        root = str(tmpdir.mkdir("anomaly_openvino_test"))

        setup_task_environment = deepcopy(setup_task_environment)  # since fixture is mutable
        task_type = setup_task_environment.task_type
        dataset: DatasetEntity = setup_task_environment.dataset
        task_environment = setup_task_environment.task_environment
        output_model = setup_task_environment.output_model

        # set normalization params for the output model
        train_task = TrainingTask(task_environment, output_path=root)
        self.set_normalization_params(output_model)
        train_task.save_model(output_model)
        task_environment.model = output_model
        train_task.export(ExportType.OPENVINO, output_model)

        # Create OpenVINO task
        openvino_task = OpenVINOTask(task_environment)

        # call inference
        dataset = dataset.get_subset(Subset.VALIDATION)
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
