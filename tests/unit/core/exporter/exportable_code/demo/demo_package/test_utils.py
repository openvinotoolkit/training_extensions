# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of demo_package utils."""

from unittest.mock import MagicMock

import pytest
from otx.core.exporter.exportable_code.demo.demo_package import utils as target_file
from otx.core.exporter.exportable_code.demo.demo_package.utils import create_visualizer, get_model_path, get_parameters


def test_get_model_path(mocker, tmp_path):
    fake_file = tmp_path / "fake_file.txt"
    ov_file = tmp_path / "openvino.xml"
    ov_file.touch()
    mocker.patch.object(target_file, "__file__", str(fake_file))

    assert ov_file == get_model_path(None)


def test_get_model_path_no_model_file(mocker, tmp_path):
    fake_file = tmp_path / "fake_file.txt"
    mocker.patch.object(target_file, "__file__", str(fake_file))

    with pytest.raises(OSError, match="model was not found"):
        get_model_path(None)


@pytest.fixture()
def mock_json(mocker):
    return mocker.patch.object(target_file, "json")


def test_get_parameters(mocker, tmp_path, mock_json):
    fake_file = tmp_path / "fake_file.txt"
    cfg_file = tmp_path / "config.json"
    cfg_file.touch()
    mocker.patch.object(target_file, "__file__", str(fake_file))

    get_parameters(None)
    mock_json.load.assert_called()


def test_get_parameters_no_cfg(mocker, tmp_path, mock_json):  # noqa: ARG001
    fake_file = tmp_path / "fake_file.txt"
    mocker.patch.object(target_file, "__file__", str(fake_file))

    with pytest.raises(OSError, match="config was not found"):
        get_parameters(None)


TASK_VISUALIZER = {
    "CLASSIFICATION": "ClassificationVisualizer",
    "DETECTION": "ObjectDetectionVisualizer",
    "SEGMENTATION": "SemanticSegmentationVisualizer",
    "INSTANCE_SEGMENTATION": "InstanceSegmentationVisualizer",
}


@pytest.mark.parametrize("task_type", TASK_VISUALIZER.keys())
def test_create_visualizer(mocker, task_type):
    mock_visualizer = mocker.patch.object(target_file, TASK_VISUALIZER[task_type])
    create_visualizer(task_type, MagicMock())
    mock_visualizer.assert_called_once()


def test_create_visualizer_not_implemented():
    with pytest.raises(NotImplementedError):
        create_visualizer("unsupported", MagicMock())
