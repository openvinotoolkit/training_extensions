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
import tempfile

import numpy as np
import pytest
from ote_anomalib.config import get_anomalib_config
from helpers.config import get_config_and_task_name
from helpers.dummy_dataset import TestDataset
from helpers.train import OTEAnomalyTrainer

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    ["task_path", "template_path"],
    [("anomaly_classification", "padim"), ("anomaly_classification", "stfpm")],
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

        ote_config, task_name = get_config_and_task_name(f"{task_path}/configs/{template_path}/template.yaml")

        # change parameter value in OTE config
        ote_config.dataset.train_batch_size = train_batch_size
        # convert OTE -> Anomalib
        anomalib_config = get_anomalib_config(task_name, ote_config)
        # check if default parameter was overwritten
        assert anomalib_config.dataset.train_batch_size == train_batch_size

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
        assert base_results.performance.score.value > 0.5

        # Convert the model to OpenVINO
        self._trainer.export()
        openvino_results = self._trainer.validate(task=self._trainer.openvino_task)

        base_probability_scores = [
            base_results.prediction_dataset[i].annotation_scene.annotations[0].get_labels()[0].probability
            for i in range(len(base_results.prediction_dataset))
        ]
        openvino_probability_scores = [
            openvino_results.prediction_dataset[i].annotation_scene.annotations[0].get_labels()[0].probability
            for i in range(len(openvino_results.prediction_dataset))
        ]

        assert np.allclose(base_probability_scores, openvino_probability_scores, rtol=0.05)

    @TestDataset(num_train=200, num_test=10, dataset_path="./datasets/MVTec", use_mvtec=False)
    def test_ote_deploy(self, task_path, template_path, dataset_path="./datasets/MVTec", category="bottle"):
        """
        E2E Test generation of exportable code.
        """
        self._trainer = OTEAnomalyTrainer(
            model_template_path=f"{task_path}/configs/{template_path}/template.yaml",
            dataset_path=dataset_path,
            category=category,
        )

        # Train is called as we need threshold
        self._trainer.train()

        # Convert the model to OpenVINO
        self._trainer.export()

        # generate exportable code
        self._trainer.deploy()

        # write zip file from the model weights
        with tempfile.TemporaryDirectory() as tempdir:
            zipfile = os.path.join(tempdir, "openvino.zip")
            with open(zipfile, "wb") as output_arch:
                output_arch.write(self._trainer.output_model.exportable_code)

            # check if size of zip is greater than 0 bytes
            assert os.path.getsize(zipfile) > 0
