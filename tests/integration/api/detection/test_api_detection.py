"""API Tests for detection training"""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os.path as osp
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from otx.algorithms.common.tasks.training_base import BaseTask
from otx.algorithms.detection.tasks import DetectionInferenceTask, DetectionTrainTask
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.metrics import Performance
from otx.api.entities.model import ModelEntity
from otx.api.entities.model_template import (
    TaskType,
    parse_model_template,
)
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.subset import Subset
from otx.api.entities.train_parameters import TrainParameters
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from tests.integration.api.detection.api_detection import DetectionTaskAPIBase
from tests.test_suite.e2e_test_system import e2e_pytest_api

DEFAULT_DET_TEMPLATE_DIR = osp.join("src/otx/algorithms/detection/configs", "detection", "mobilenetv2_atss")


def task_eval(task: BaseTask, model: ModelEntity, dataset: DatasetEntity) -> Performance:
    start_time = time.time()
    result_dataset = task.infer(dataset.with_empty_annotations())
    end_time = time.time()
    print(f"{len(dataset)} analysed in {end_time - start_time} seconds")
    result_set = ResultSetEntity(model=model, ground_truth_dataset=dataset, prediction_dataset=result_dataset)
    task.evaluate(result_set)
    assert result_set.performance is not None
    return result_set.performance


class TestDetectionTaskAPI(DetectionTaskAPIBase):
    """
    Collection of tests for OTX API and OTX Model Templates
    """

    @e2e_pytest_api
    def test_reading_detection_model_template(self):
        detection_template = ["mobilenetv2_atss"]
        for model_template in detection_template:
            parse_model_template(
                osp.join("src/otx/algorithms/detection/configs", "detection", model_template, "template.yaml")
            )

    @e2e_pytest_api
    def test_cancel_training_detection(self):
        """
        Tests starting and cancelling training.

        Flow of the test:
        - Creates a randomly annotated project with a small dataset containing 3 classes:
            ['rectangle', 'triangle', 'circle'].
        - Start training and give cancel training signal after 10 seconds. Assert that training
            stops within 35 seconds after that
        - Start training and give cancel signal immediately. Assert that training stops within 25 seconds.

        This test should be finished in under one minute on a workstation.
        """
        hyper_parameters, model_template = self.setup_configurable_parameters(DEFAULT_DET_TEMPLATE_DIR, num_iters=500)
        detection_environment, dataset = self.init_environment(hyper_parameters, model_template, 64)

        detection_task = DetectionTrainTask(task_environment=detection_environment)

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="train_thread")

        output_model = ModelEntity(
            dataset,
            detection_environment.get_model_configuration(),
        )

        training_progress_curve = []

        def progress_callback(progress: float, score: Optional[float] = None):
            training_progress_curve.append(progress)

        train_parameters = TrainParameters()
        train_parameters.update_progress = progress_callback

        # Test stopping after some time
        start_time = time.time()
        train_future = executor.submit(detection_task.train, dataset, output_model, train_parameters)
        # give train_thread some time to initialize the model
        while not detection_task._is_training:
            time.sleep(10)
        detection_task.cancel_training()

        # stopping process has to happen in less than 35 seconds
        train_future.result()
        assert training_progress_curve[-1] == 100
        assert time.time() - start_time < 100, "Expected to stop within 100 seconds."

        # Test stopping immediately
        start_time = time.time()
        train_future = executor.submit(detection_task.train, dataset, output_model)
        detection_task.cancel_training()

        train_future.result()
        assert time.time() - start_time < 25  # stopping process has to happen in less than 25 seconds

    @e2e_pytest_api
    def test_training_progress_tracking(self):
        hyper_parameters, model_template = self.setup_configurable_parameters(DEFAULT_DET_TEMPLATE_DIR, num_iters=5)
        detection_environment, dataset = self.init_environment(hyper_parameters, model_template, 50)

        task = DetectionTrainTask(task_environment=detection_environment)
        print("Task initialized, model training starts.")

        training_progress_curve = []

        def progress_callback(progress: float, score: Optional[float] = None):
            training_progress_curve.append(progress)

        train_parameters = TrainParameters()
        train_parameters.update_progress = progress_callback
        output_model = ModelEntity(
            dataset,
            detection_environment.get_model_configuration(),
        )
        task.train(dataset, output_model, train_parameters)

        assert len(training_progress_curve) > 0
        assert np.all(training_progress_curve[1:] >= training_progress_curve[:-1])

    @e2e_pytest_api
    def test_inference_progress_tracking(self):
        hyper_parameters, model_template = self.setup_configurable_parameters(DEFAULT_DET_TEMPLATE_DIR, num_iters=10)
        detection_environment, dataset = self.init_environment(hyper_parameters, model_template, 50)

        task = DetectionInferenceTask(task_environment=detection_environment)
        print("Task initialized, model inference starts.")
        inference_progress_curve = []

        def progress_callback(progress: int):
            assert isinstance(progress, int)
            inference_progress_curve.append(progress)

        inference_parameters = InferenceParameters()
        inference_parameters.update_progress = progress_callback
        task.infer(dataset, inference_parameters)

        assert len(inference_progress_curve) > 0
        assert np.all(inference_progress_curve[1:] >= inference_progress_curve[:-1])

    @e2e_pytest_api
    def test_inference_task(self):
        # Prepare pretrained weights
        hyper_parameters, model_template = self.setup_configurable_parameters(DEFAULT_DET_TEMPLATE_DIR, num_iters=2)
        detection_environment, dataset = self.init_environment(hyper_parameters, model_template, 50)
        val_dataset = dataset.get_subset(Subset.VALIDATION)

        train_task = DetectionTrainTask(task_environment=detection_environment)

        training_progress_curve = []

        def progress_callback(progress: float, score: Optional[float] = None):
            training_progress_curve.append(progress)

        train_parameters = TrainParameters()
        train_parameters.update_progress = progress_callback
        trained_model = ModelEntity(
            dataset,
            detection_environment.get_model_configuration(),
        )
        train_task.train(dataset, trained_model, train_parameters)
        performance_after_train = task_eval(train_task, trained_model, val_dataset)

        # Create InferenceTask
        detection_environment.model = trained_model
        inference_task = DetectionInferenceTask(task_environment=detection_environment)

        performance_after_load = task_eval(inference_task, trained_model, val_dataset)

        assert performance_after_train == performance_after_load

        # Export
        exported_model = ModelEntity(dataset, detection_environment.get_model_configuration())
        inference_task.export(ExportType.OPENVINO, exported_model)
