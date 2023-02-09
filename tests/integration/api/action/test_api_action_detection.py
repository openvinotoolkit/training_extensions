"""API Tests for Action Detection training"""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import glob
import os.path as osp
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import pytest

from otx.algorithms.action.tasks import ActionInferenceTask, ActionTrainTask
from otx.algorithms.common.tasks.training_base import BaseTask
from otx.api.configuration.helper import create
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.metrics import Performance
from otx.api.entities.model import ModelEntity
from otx.api.entities.model_template import parse_model_template
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.entities.train_parameters import TrainParameters
from otx.core.data.adapter import get_dataset_adapter
from tests.test_suite.e2e_test_system import e2e_pytest_api

DEFAULT_ACTION_TEMPLATE_DIR = osp.join("otx/algorithms/action/configs", "detection", "x3d_fast_rcnn")


def task_eval(task: BaseTask, model: ModelEntity, dataset: DatasetEntity) -> Performance:
    start_time = time.time()
    result_dataset = task.infer(dataset.with_empty_annotations())
    end_time = time.time()
    print(f"{len(dataset)} analysed in {end_time - start_time} seconds")
    result_set = ResultSetEntity(model=model, ground_truth_dataset=dataset, prediction_dataset=result_dataset)
    task.evaluate(result_set)
    assert result_set.performance is not None
    return result_set.performance


class TestActionTaskAPI:
    """
    Collection of tests for OTX API and OTX Model Templates
    """

    train_data_roots = "tests/assets/cvat_dataset/action_detection/train"
    val_data_roots = "tests/assets/cvat_dataset/action_detection/train"

    @e2e_pytest_api
    def test_reading_action_model_template(self):
        model_templates = ["x3d_fast_rcnn"]
        for model_template in model_templates:
            parse_model_template(
                osp.join("otx/algorithms/action/configs", "detection", model_template, "template.yaml")
            )

    def init_environment(self, params, model_template):
        dataset_adapter = get_dataset_adapter(
            model_template.task_type,
            train_data_roots=self.train_data_roots,
            val_data_roots=self.val_data_roots,
        )
        dataset = dataset_adapter.get_otx_dataset()
        label_schema = dataset_adapter.get_label_schema()
        environment = TaskEnvironment(
            model=None,
            hyper_parameters=params,
            label_schema=label_schema,
            model_template=model_template,
        )
        return environment, dataset

    @staticmethod
    def setup_configurable_parameters(template_dir, num_iters=10):
        glb = glob.glob(f"{template_dir}/template*.yaml")
        template_path = glb[0] if glb else None
        if not template_path:
            raise RuntimeError(f"Template YAML not found: {template_dir}")

        model_template = parse_model_template(template_path)
        hyper_parameters = create(model_template.hyper_parameters.data)
        hyper_parameters.learning_parameters.num_iters = num_iters
        hyper_parameters.postprocessing.result_based_confidence_threshold = False
        hyper_parameters.postprocessing.confidence_threshold = 0.1
        return hyper_parameters, model_template

    @e2e_pytest_api
    @pytest.mark.skip(reason="mmaction does not support EpochRunnerWithCancel")
    def test_cancel_training_action(self):
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
        hyper_parameters, model_template = self.setup_configurable_parameters(
            DEFAULT_ACTION_TEMPLATE_DIR, num_iters=500
        )
        action_environment, dataset = self.init_environment(hyper_parameters, model_template)

        action_task = ActionTrainTask(task_environment=action_environment)

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="train_thread")

        output_model = ModelEntity(
            dataset,
            action_environment.get_model_configuration(),
        )

        training_progress_curve = []

        def progress_callback(progress: float, score: Optional[float] = None):
            training_progress_curve.append(progress)

        train_parameters = TrainParameters()
        train_parameters.update_progress = progress_callback

        # Test stopping after some time
        start_time = time.time()
        train_future = executor.submit(action_task.train, dataset, output_model, train_parameters)
        # give train_thread some time to initialize the model
        while not action_task._is_training:
            time.sleep(10)
        action_task.cancel_training()

        # stopping process has to happen in less than 35 seconds
        train_future.result()
        assert training_progress_curve[-1] == 100
        assert time.time() - start_time < 100, "Expected to stop within 100 seconds."

        # Test stopping immediately
        start_time = time.time()
        train_future = executor.submit(action_task.train, dataset, output_model)
        action_task.cancel_training()

        train_future.result()
        assert time.time() - start_time < 25  # stopping process has to happen in less than 25 seconds

    @e2e_pytest_api
    def test_training_progress_tracking(self):
        hyper_parameters, model_template = self.setup_configurable_parameters(DEFAULT_ACTION_TEMPLATE_DIR, num_iters=5)
        action_environment, dataset = self.init_environment(hyper_parameters, model_template)
        task = ActionTrainTask(task_environment=action_environment)
        print("Task initialized, model training starts.")

        training_progress_curve = []

        def progress_callback(progress: float, score: Optional[float] = None):
            training_progress_curve.append(progress)

        train_parameters = TrainParameters()
        train_parameters.update_progress = progress_callback
        output_model = ModelEntity(
            dataset,
            action_environment.get_model_configuration(),
        )
        task.train(dataset, output_model, train_parameters)

        assert len(training_progress_curve) > 0
        assert np.all(training_progress_curve[1:] >= training_progress_curve[:-1])

    @e2e_pytest_api
    def test_inference_progress_tracking(self):
        hyper_parameters, model_template = self.setup_configurable_parameters(DEFAULT_ACTION_TEMPLATE_DIR, num_iters=10)
        action_environment, dataset = self.init_environment(hyper_parameters, model_template)

        task = ActionInferenceTask(task_environment=action_environment)
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
        # FIXME CVS-103071 will handle this
        # Prepare pretrained weights
        hyper_parameters, model_template = self.setup_configurable_parameters(DEFAULT_ACTION_TEMPLATE_DIR, num_iters=2)
        action_environment, dataset = self.init_environment(hyper_parameters, model_template)
        # val_dataset = dataset.get_subset(Subset.VALIDATION)

        train_task = ActionTrainTask(task_environment=action_environment)

        training_progress_curve = []

        def progress_callback(progress: float, score: Optional[float] = None):
            training_progress_curve.append(progress)

        train_parameters = TrainParameters()
        train_parameters.update_progress = progress_callback
        trained_model = ModelEntity(
            dataset,
            action_environment.get_model_configuration(),
        )
        train_task.train(dataset, trained_model, train_parameters)
        # performance_after_train = task_eval(train_task, trained_model, val_dataset)

        # Create InferenceTask
        # action_environment.model = trained_model
        # inference_task = ActionInferenceTask(task_environment=action_environment)

        # performance_after_load = task_eval(inference_task, trained_model, val_dataset)

        # FIXME CVS-103071 will handle this
        # assert performance_after_train == performance_after_load

        # Export
        # CVS-102941 ONNX export of action detection model keeps failed
        # exported_model = ModelEntity(dataset, action_environment.get_model_configuration())
        # inference_task.export(ExportType.OPENVINO, exported_model)
