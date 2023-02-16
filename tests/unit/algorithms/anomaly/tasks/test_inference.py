"""Tests the methods in the Inference task."""

# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

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
    """Tests the methods in the inference task."""

    def test_inference(self, tmpdir, setup_task_environment):
        """Tests the inference method."""
        root = str(tmpdir.mkdir("anomaly_inference_test"))

        # Get task environment
        setup_task_environment = deepcopy(setup_task_environment)  # since fixture is mutable
        task_environment = setup_task_environment.task_environment
        task_type = setup_task_environment.task_type
        output_model = setup_task_environment.output_model
        dataset = setup_task_environment.dataset

        # 1. Create the training task and get inference results on an untrained model.
        dataset = dataset.get_subset(Subset.VALIDATION)
        train_task = TrainingTask(task_environment, output_path=root)
        dataset = train_task.infer(dataset.with_empty_annotations(), InferenceParameters(is_evaluation=True))
        train_task.save_model(output_model)
        # 2. check if the model is saved correctly
        assert output_model.get_data("weights.pth") is not None  # Should not raise an error

        # 3. Create new task environment and inference task and test inference
        new_dataset: DatasetEntity = get_shapes_dataset(task_type, one_each=True)
        gt_val_dataset = new_dataset.get_subset(Subset.VALIDATION)
        new_task_environment = create_task_environment(gt_val_dataset, task_type)
        # this loads the output model from the previous training task when creating the new InferenceTask
        new_task_environment.model = output_model
        inference_task = InferenceTask(new_task_environment, output_path=root)
        pred_val_dataset = inference_task.infer(
            gt_val_dataset.with_empty_annotations(), InferenceParameters(is_evaluation=True)
        )

        # compare labels with the original validation dataset
        for item1, item2 in zip(dataset, pred_val_dataset):
            assert set([label.name for label in item1.annotation_scene.get_labels()]) == set(
                [label.name for label in item2.annotation_scene.get_labels()]
            )

        # 4. Check whether performance metrics are produced correctly
        # This tests whether the performance metrics are calculated correctly and assigned to the result set
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

        # 5. Check if OpenVINO model can be generated
        inference_task.export(ExportType.OPENVINO, output_model)
        assert output_model.get_data("openvino.bin") is not None  # Should not raise an error
