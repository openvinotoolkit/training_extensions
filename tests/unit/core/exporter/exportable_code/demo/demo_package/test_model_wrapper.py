# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of demo_package model_wrapper."""

from unittest.mock import MagicMock

import pytest

target_file = None
ModelWrapper, TaskType = None, None


@pytest.fixture(scope="module", autouse=True)
def fxt_import_module():
    global target_file  # noqa: PLW0603
    global ModelWrapper, TaskType
    from otx.core.exporter.exportable_code.demo.demo_package import model_wrapper
    from otx.core.exporter.exportable_code.demo.demo_package.model_wrapper import ModelWrapper as Cls1
    from otx.core.exporter.exportable_code.demo.demo_package.model_wrapper import TaskType as Cls2

    target_file = model_wrapper
    ModelWrapper, TaskType = Cls1, Cls2


class TestModelWrapper:
    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        mocker.patch.object(target_file, "OpenvinoAdapter")
        mocker.patch.object(target_file, "create_core")
        mocker.patch.object(
            target_file,
            "get_parameters",
            return_value={
                "model_parameters": {"labels": "label"},
                "task_type": "CLASSIFICATION",
                "model_type": "type_of_model",
            },
        )
        mocker.patch.object(target_file, "get_model_path")
        self.mock_core_model = MagicMock()
        mocker.patch.object(target_file, "Model").create_model.return_value = self.mock_core_model

    @pytest.fixture()
    def model_dir(self, tmp_path):
        (tmp_path / "config.json").touch()
        return tmp_path

    def test_init(self, model_dir):
        ModelWrapper(model_dir)

    @pytest.fixture()
    def model_wrapper(self, model_dir):
        return ModelWrapper(model_dir)

    def test_task_type(self, model_wrapper):
        assert model_wrapper.task_type == TaskType.CLASSIFICATION

    def test_labels(self, model_wrapper):
        assert model_wrapper.labels == "label"

    def test_infer(self, model_wrapper):
        frame = MagicMock()
        pred, frame_meta = model_wrapper.infer(frame)

        self.mock_core_model.assert_called_once_with(frame)
        assert pred == self.mock_core_model.return_value
        assert frame_meta == {"original_shape": frame.shape}

    def test_call(self, mocker, model_wrapper):
        input_data = MagicMock()
        spy_infer = mocker.spy(model_wrapper, "infer")
        model_wrapper(input_data)
        spy_infer.assert_called_once_with(input_data)
