"""Tests the methods in the Inference task."""

# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from otx.algorithms.anomaly.tasks.inference import InferenceTask
from otx.algorithms.anomaly.tasks.train import TrainingTask
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.model import ModelEntity
from otx.api.entities.model_template import TaskType
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.subset import Subset
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from tests.unit.algorithms.anomaly.helpers.dummy_dataset import get_shapes_dataset
from tests.unit.algorithms.anomaly.helpers.utils import create_task_environment


class TestInferenceTask:
    """Tests the method in the inference task."""

    @pytest.mark.parametrize(
        "task_type", [TaskType.ANOMALY_CLASSIFICATION, TaskType.ANOMALY_DETECTION, TaskType.ANOMALY_SEGMENTATION]
    )
    def test_inference(self, task_type, tmpdir):
        """Tests the inference method."""
        root = str(tmpdir.mkdir("anomaly_inference_test"))
        dataset: DatasetEntity = get_shapes_dataset(task_type, one_each=True)
        dataset = dataset.get_subset(Subset.TESTING)
        task_environment = create_task_environment(dataset, task_type)
        # get model configuration
        output_model = ModelEntity(
            dataset,
            task_environment.get_model_configuration(),
        )
        train_task = TrainingTask(task_environment, output_path=root)
        dataset = train_task.infer(dataset.with_empty_annotations(), InferenceParameters(is_evaluation=True))
        train_task.save_model(output_model)

        new_dataset: DatasetEntity = get_shapes_dataset(task_type, one_each=True)
        gt_val_dataset = new_dataset.get_subset(Subset.VALIDATION)
        new_task_environment = create_task_environment(gt_val_dataset, task_type)
        new_task_environment.model = output_model
        inference_task = InferenceTask(new_task_environment, output_path=root)
        pred_val_dataset = inference_task.infer(
            gt_val_dataset.with_empty_annotations(), InferenceParameters(is_evaluation=True)
        )

        # check that the labels are the same
        for item1, item2 in zip(dataset, pred_val_dataset):
            assert set([label.name for label in item1.annotation_scene.get_labels()]) == set(
                [label.name for label in item2.annotation_scene.get_labels()]
            )

        # check whether performance metrics are produced correctly
        output_model = ModelEntity(
            gt_val_dataset,
            new_task_environment.get_model_configuration(),
        )
        result_set = ResultSetEntity(
            model=output_model, ground_truth_dataset=gt_val_dataset, prediction_dataset=pred_val_dataset
        )
        inference_task.evaluate(result_set)
        if task_type in (TaskType.ANOMALY_CLASSIFICATION, TaskType.ANOMALY_DETECTION):
            assert result_set.performance.score.name == "f-measure"
        elif task_type == TaskType.ANOMALY_SEGMENTATION:
            assert result_set.performance.score.name == "Dice Average"

        # check if OpenVINO model is created
        inference_task.export(ExportType.OPENVINO, output_model)
        output_model.get_data("openvino.bin")  # Should not raise an error
