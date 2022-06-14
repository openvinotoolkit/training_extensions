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
from ote_anomalib.configs import get_anomalib_config
from tools.sample import OteAnomalyTask
from tests.helpers.config import get_config_and_task_name
from tests.helpers.dummy_dataset import TestDataset

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    ["task_path", "template_path"],
    [
        ("anomaly_classification", "padim"),
        ("anomaly_classification", "stfpm"),
        ("anomaly_segmentation", "padim"),
        ("anomaly_segmentation", "stfpm"),
    ],
)
class TestAnomalyClassification:
    """
    Anomaly Classification Task Tests.
    """

    _trainer: OteAnomalyTask

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
        self,
        task_path,
        template_path,
        dataset_path="./datasets/MVTec",
        category="bottle",
    ):
        """
        E2E Train-Export Should Yield Similar Inference Results
        """
        # Train the model
        dataset_path = os.path.join(dataset_path, category)
        self._trainer = OteAnomalyTask(
            dataset_path=dataset_path,
            model_template_path=f"{task_path}/configs/{template_path}/template.yaml",
            seed=0,
        )
        output_model = self._trainer.train()

        base_results = self._trainer.infer(task=self._trainer.torch_task, output_model=output_model)
        self._trainer.evaluate(task=self._trainer.torch_task, result_set=base_results)

        if task_path == "anomaly_classification":  # skip this check for anomaly segmentation until we switch metrics
            assert base_results.performance.score.value > 0.5

        # Convert the model to OpenVINO
        self._trainer.export()
        openvino_results = self._trainer.infer(task=self._trainer.openvino_task, output_model=output_model)
        self._trainer.evaluate(task=self._trainer.openvino_task, result_set=openvino_results)

        assert np.allclose(
            base_results.performance.score.value,
            openvino_results.performance.score.value,
            atol=0.1,
        )

        # NNCF optimization
        self._trainer.optimize_nncf()

        base_nncf_results = self._trainer.infer(task=self._trainer.torch_task, output_model=output_model)
        self._trainer.evaluate(task=self._trainer.torch_task, result_set=base_nncf_results)
        if task_path == "anomaly_classification":  # skip this check for anomaly segmentation until we switch metrics
            assert base_nncf_results.performance.score.value > 0.5

        self._trainer.export_nncf()
        openvino_results = self._trainer.infer(task=self._trainer.openvino_task, output_model=output_model)
        self._trainer.evaluate(task=self._trainer.openvino_task, result_set=openvino_results)
        assert np.allclose(
            base_nncf_results.performance.score.value,
            openvino_results.performance.score.value,
            atol=0.2,
        )

    @TestDataset(num_train=200, num_test=10, dataset_path="./datasets/MVTec", use_mvtec=False)
    def test_ote_deploy(
        self,
        task_path,
        template_path,
        dataset_path="./datasets/MVTec",
        category="bottle",
    ):
        """
        E2E Test generation of exportable code.
        """
        dataset_path = os.path.join(dataset_path, category)
        self._trainer = OteAnomalyTask(
            model_template_path=f"{task_path}/configs/{template_path}/template.yaml",
            dataset_path=dataset_path,
            seed=0,
        )

        # Train is called as we need threshold
        self._trainer.train()

        # Convert the model to OpenVINO
        output_model = self._trainer.export()

        # generate exportable code
        try:
            output_model.get_data("openvino.bin")
        except KeyError as error:
            raise KeyError(
                "Could not get `openvino.bin` from model. Make sure that the model is exported to OpenVINO first"
            ) from error

        self._trainer.openvino_task.deploy(output_model)

        # write zip file from the model weights
        with tempfile.TemporaryDirectory() as tempdir:
            zipfile = os.path.join(tempdir, "openvino.zip")
            with open(zipfile, "wb") as output_arch:
                output_arch.write(output_model.exportable_code)

            # check if size of zip is greater than 0 bytes
            assert os.path.getsize(zipfile) > 0
