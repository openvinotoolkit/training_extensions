"""Tests the methods in the OpenVINO task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
from otx.algorithms.visual_prompting.tasks.openvino import OpenVINOTask
from tests.unit.algorithms.visual_prompting.test_helpers import (
    init_environment,
)
from addict import Dict as ADDict
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestOpenVINOTask:
    @pytest.fixture
    def load_openvino_task(self, mocker):
        """Load the OpenVINOTask."""
        def _load_openvino_task():
            mocker_model = mocker.patch("otx.api.entities.model.ModelEntity")
            self.task_environment = init_environment(mocker_model)
            return OpenVINOTask(task_environment=self.task_environment)
        return _load_openvino_task

    @e2e_pytest_unit    
    def test_get_config(self, mocker, load_openvino_task):
        """Test get_config."""
        mocker.patch("otx.algorithms.visual_prompting.tasks.openvino.OpenVINOTask.load_model")

        openvino_task = load_openvino_task()

        assert openvino_task.mode == "openvino"
        assert isinstance(openvino_task.config, ADDict)
        assert openvino_task.config.dataset.task == "visual_prompting"

    @e2e_pytest_unit
    def test_load_model(self, mocker, load_openvino_task):
        """Test load_model."""
        
        mocker_ie_core = mocker.patch("otx.algorithms.visual_prompting.tasks.openvino.IECore")
        mocker.patch("otx.algorithms.visual_prompting.tasks.openvino.OpenVINOTask.get_config")

        openvino_task = load_openvino_task()

        assert openvino_task.mode == "openvino"
        mocker_ie_core.assert_called_once()
        assert hasattr(openvino_task, "sam_image_encoder")
        assert hasattr(openvino_task, "sam_decoder")

    @e2e_pytest_unit
    def test_infer(self):
        pass

