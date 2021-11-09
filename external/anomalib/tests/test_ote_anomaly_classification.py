"""
Test Anomaly Classification Task
"""

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
import logging
import os
from threading import Thread

import numpy as np
import pytest
from ote_sdk.entities.model import ModelEntity
from ote_sdk.entities.model import ModelStatus
from ote_sdk.entities.train_parameters import TrainParameters

from core.config import get_anomalib_config
from tests.helpers.config import get_config
from tests.helpers.dataset import OTEAnomalyDatasetGenerator
from tests.helpers.dummy_dataset import GeneratedDummyDataset
from tests.helpers.dummy_dataset import TestDataset
from tests.helpers.train import OTEAnomalyTrainer

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    ["task_path", "template_path"], [("anomaly_classification", "padim"), ("anomaly_classification", "stfpm")]
)
class TestAnomalyClassification:
    """
    Anomaly Classification Task Tests.
    """

    _trainer: OTEAnomalyTrainer

    @staticmethod
    def test_ote_config(task_path, template_path):
        """
        Test generation of OTE config object from model template and conversion to Anomalib format. Also checks if
        default values are overwritten in Anomalib config.
        """
        train_batch_size = 16

        ote_config = get_config(f"{task_path}/configs/{template_path}/template.yaml")

        # change parameter value in OTE config
        ote_config.dataset.train_batch_size = train_batch_size
        # convert OTE -> Anomalib
        anomalib_config = get_anomalib_config(ote_config)
        # check if default parameter was overwritten
        assert anomalib_config.dataset.train_batch_size == train_batch_size

    @TestDataset(num_train=200, num_test=50, dataset_path="./datasets/MVTec", use_mvtec=False)
    def test_cancel_training(self, task_path, template_path, dataset_path="./datasets/MVTec", category="bottle"):
        """
        Training should stop when `cancel_training` is called
        """
        self._trainer = OTEAnomalyTrainer(
            model_template_path=f"{task_path}/configs/{template_path}/template.yaml",
            dataset_path=dataset_path,
            category=category,
        )
        thread = Thread(target=self._trainer.train)
        thread.start()
        self._trainer.cancel_training()
        assert self._trainer.base_task.model.results.performance == {}

    @TestDataset(num_train=200, num_test=10, dataset_path="./datasets/MVTec", use_mvtec=False)
    def test_ote_train_export_and_optimize(
        self, task_path, template_path, dataset_path="./datasets/MVTec", category="bottle"
    ):
        """
        E2E Train-Export Should Yield Similar Inference Results
        """
        # Train the model
        self._trainer = OTEAnomalyTrainer(
            model_template_path=f"{task_path}/configs/{template_path}/template.yaml",
            dataset_path=dataset_path,
            category=category,
        )
        self._trainer.train()
        base_results = self._trainer.validate(task=self._trainer.base_task)

        # Convert the model to OpenVINO
        self._trainer.export()
        openvino_results = self._trainer.validate(task=self._trainer.openvino_task)

        # Optimize the OpenVINO Model via POT
        optimized_openvino_results = self._trainer.validate(task=self._trainer.openvino_task, optimize=True)

        # Performance should be higher than a threshold.
        assert base_results.performance.score.value > 0.6

        # Performance should be almost the same
        assert np.allclose(base_results.performance.score.value, openvino_results.performance.score.value)
        assert np.allclose(openvino_results.performance.score.value, optimized_openvino_results.performance.score.value)


class TestModelMonitor:

    _trainer: OTEAnomalyTrainer

    def test_model_monitor(self, dataset_path="./datasets/MVTec", category="bottle"):
        """
        Test if model monitor callback produces the correct model status
        """
        # Create task and task environment
        trainer = OTEAnomalyTrainer(
            model_template_path=f"anomaly_classification/configs/padim/template.yaml",
            dataset_path=dataset_path,
            category=category,
        )
        task = trainer.base_task
        task_environment = task.task_environment

        # create output model
        output_model = ModelEntity(
            train_dataset=trainer.dataset,
            configuration=task_environment.get_model_configuration(),
            model_status=ModelStatus.NOT_READY,
        )

        # create dataset
        with GeneratedDummyDataset(num_train=200,
                                   num_test=10,
                                   train_shapes=["triangle"],
                                   test_shapes=["star"]) as dataset_path:
            dataset_generator = OTEAnomalyDatasetGenerator(path=os.path.join(dataset_path, "shapes"))
            dataset = dataset_generator.generate()

            # initial training should return a trained model
            task.train(dataset=dataset, output_model=output_model, train_parameters=TrainParameters())
            assert task.task_environment.model.model_status == ModelStatus.SUCCESS

        # switch normal and abnormal categories
        with GeneratedDummyDataset(num_train=200,
                                   num_test=10,
                                   train_shapes=["star"],
                                   test_shapes=["triangle"]) as dataset_path:
            dataset_generator = OTEAnomalyDatasetGenerator(path=os.path.join(dataset_path, "shapes"))
            dataset = dataset_generator.generate()

            # training should improve on the swapped dataset
            task.train(dataset=dataset, output_model=output_model, train_parameters=TrainParameters())
            assert task.task_environment.model.model_status == ModelStatus.SUCCESS

            # training again on the same dataset should not return a new model
            task.train(dataset=dataset, output_model=output_model, train_parameters=TrainParameters())
            assert task.task_environment.model.model_status == ModelStatus.NOT_IMPROVED
