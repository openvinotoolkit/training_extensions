# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import glob
import io
import os.path as osp
import random
import time
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor
from subprocess import run  # nosec
from typing import Optional
import sys

import numpy as np
import torch
from bson import ObjectId
from ote_sdk.test_suite.e2e_test_system import e2e_pytest_api
from ote_sdk.configuration.helper import convert, create
from ote_sdk.entities.annotation import AnnotationSceneEntity, AnnotationSceneKind
from ote_sdk.entities.dataset_item import DatasetItemEntity
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.image import Image
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.model_template import TaskType, task_type_to_label_domain
from ote_sdk.entities.metrics import Performance
from ote_sdk.entities.model import ModelEntity, ModelFormat, ModelOptimizationType
from ote_sdk.entities.model_template import parse_model_template
from ote_sdk.entities.optimization_parameters import OptimizationParameters
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.train_parameters import TrainParameters
from ote_sdk.tests.test_helpers import generate_random_annotated_image
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType, IExportTask
from ote_sdk.usecases.tasks.interfaces.optimization_interface import OptimizationType
from ote_sdk.utils.shape_factory import ShapeFactory

from ote_cli.datasets import get_dataset_class
from speech_to_text.ote import OTESpeechToTextDataset, OTESpeechToTextTaskParameters, OpenVINOSpeechToTextTask, OTESpeechToTextTaskTrain, OpenVINOSpeechToTextTask

DEFAULT_TEMPLATE_DIR = osp.join('configs', 'ote', 'quartznet')

class API(unittest.TestCase):
    """
    Collection of tests for OTE API and OTE Model Templates
    """

    def init_environment(
            self,
            params,
            model_template,
            task_type=TaskType.SPEECH_TO_TEXT):

        environment = TaskEnvironment(model=None, hyper_parameters=params, label_schema=None,
                                      model_template=model_template)

        pretrained_model = ModelEntity(
            None,
            environment.get_model_configuration(),
        )

        pretrained_model.set_data(
            "weights.ckpt",
            open(osp.join('tests', 'testdata', 'weights.ckpt'), "rb").read()
        )

        environment.model = pretrained_model

        dataset = get_dataset_class(task_type)(
            train_subset={"data_root": [osp.join('tests', 'testdata', 'LibriSpeech', 'dev-clean')]},
            val_subset={"data_root": [osp.join('tests', 'testdata', 'LibriSpeech', 'dev-clean')]}
        )
        return environment, dataset

    def setup_configurable_parameters(
            self,
            template_dir,
            num_epochs=1,
            batch_size=1,
            learning_rate=5e-4,
            learning_rate_warmup_steps=0,
            lr_scheduler=False
    ):
        glb = glob.glob(f'{template_dir}/template*.yaml')
        template_path = glb[0] if glb else None
        if not template_path:
          raise RuntimeError(f"Template YAML not found: {template_dir}")

        model_template = parse_model_template(template_path)
        hyper_parameters = create(model_template.hyper_parameters.data)
        hyper_parameters.learning_parameters.num_epochs = num_epochs
        hyper_parameters.learning_parameters.batch_size = batch_size
        hyper_parameters.learning_parameters.learning_rate = learning_rate
        hyper_parameters.learning_parameters.learning_rate_warmup_steps = learning_rate_warmup_steps
        hyper_parameters.learning_parameters.lr_scheduler = lr_scheduler
        return hyper_parameters, model_template

    @e2e_pytest_api
    def test_cancel_training(self):
        """
        Tests starting and cancelling training.

        Flow of the test:
        - Load testdata.
        - Start training and give cancel training signal after 10 seconds. Assert that training
            stops within 35 seconds after that
        - Start training and give cancel signal immediately. Assert that training stops within 25 seconds.

        This test should be finished in under one minute on a workstation.
        """
        hyper_parameters, model_template = self.setup_configurable_parameters(DEFAULT_TEMPLATE_DIR, num_epochs=2)
        task_environment, dataset = self.init_environment(hyper_parameters, model_template)

        train_task = OTESpeechToTextTaskTrain(task_environment=task_environment)

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='train_thread')

        output_model = ModelEntity(
            dataset,
            task_environment.get_model_configuration(),
        )

        train_parameters = TrainParameters
        # train_parameters.update_progress = progress_callback

        # Test stopping after some time
        start_time = time.time()
        train_future = executor.submit(train_task.train, dataset, output_model, train_parameters)
        # give train_thread some time to initialize the model
        time.sleep(10)
        train_task.cancel_training()

        # stopping process has to happen in less than 35 seconds
        train_future.result()
        self.assertLess(time.time() - start_time, 100, 'Expected to stop within 100 seconds.')

    @e2e_pytest_api
    def test_task_train_and_inference(self):
        # Prepare pretrained weights
        hyper_parameters, model_template = self.setup_configurable_parameters(DEFAULT_TEMPLATE_DIR, num_epochs=2)
        task_environment, dataset = self.init_environment(hyper_parameters, model_template)
        val_dataset = dataset.get_subset(Subset.VALIDATION)

        train_task = OTESpeechToTextTaskTrain(task_environment=task_environment)
        self.addCleanup(train_task._delete_scratch_space)

        trained_model = ModelEntity(
            dataset,
            task_environment.get_model_configuration(),
        )
        train_task.train(dataset, trained_model, TrainParameters)
        performance_after_train = self.eval(train_task, trained_model, val_dataset)

        # Export model
        exported_model = ModelEntity(
            dataset,
            task_environment.get_model_configuration()
        )
        train_task.export(ExportType.OPENVINO, exported_model)

        # Create InferenceTask
        task_environment.model = exported_model
        inference_task = OpenVINOSpeechToTextTask(task_environment=task_environment)
        self.addCleanup(inference_task._delete_scratch_space)
        performance_after_load = self.eval(inference_task, exported_model, val_dataset)

        self.assertAlmostEqual(performance_after_train, performance_after_load, None, None, 0.01)

    @staticmethod
    def eval(task, model, dataset):
        predicted_dataset = task.infer(
            dataset.with_empty_annotations(),
            InferenceParameters(is_evaluation=True)
        )

        resultset = ResultSetEntity(
            model=model,
            ground_truth_dataset=dataset,
            prediction_dataset=predicted_dataset,
        )
        task.evaluate(resultset)
        return resultset.performance
