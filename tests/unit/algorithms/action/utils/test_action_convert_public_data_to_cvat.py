"""Unit Test for otx.algorithms.action.utils.convert_public_data_to_cvat."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import tempfile

import numpy as np
import pytest

from otx.algorithms.action.utils.convert_public_data_to_cvat import (
    convert_action_cls_dataset_to_datumaro,
    convert_ava_dataset_to_datumaro,
    generate_default_cvat_xml_fields,
    main,
    read_ava_csv,
    rename_and_copy,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class MockPath:
    """Mock class for pathlib.Path"""

    def __init__(self, path):
        self.path = path

    def mkdir(self, *args, **kwargs):
        return self.path


class MockFileObject:
    """Mock class for python default File object."""

    def readlines(self):
        lines = [
            "# Some comment\n",
            "video_dir frame_len class_idx\n",
            "video_dir frame_len class_idx\n",
            "video_dir frame_len class_idx\n",
            "video_dir frame_len class_idx\n",
            "video_dir frame_len class_idx\n",
        ]
        return lines

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        return None


class MockCsvFile:
    """Mock class for python default File object."""

    def __init__(self, *args, **kwargs):
        self.lines = [
            "0,0,0,0,1,1,0",
            "0,1,0,0,1,1,0",
            "0,2,0,0,1,1,0",
            "0,3,0,0,1,1,0",
            "0,4,0,0,1,1,0",
        ]

    def __enter__(self):
        return self.lines

    def __exit__(self, *args, **kwargs):
        return None


class MockElementTree:
    """Mock class for lxml.etree.ElementTree."""

    def write(self, *args, **kwargs):
        pass


@e2e_pytest_unit
def test_generate_default_cvat_xml_fields(mocker) -> None:
    """Test generate_default_cvat_xml_fields function."""
    video_path = "dummy_path"
    frame_list = ["dummy_frame0", "dummy_frame1", "dummy_frame2", "dummy_frame3"]

    mocker.patch(
        "otx.algorithms.action.utils.convert_public_data_to_cvat.cv2.imread", return_value=np.ndarray((256, 256, 3))
    )

    output = generate_default_cvat_xml_fields(1, video_path, frame_list)
    assert len(output[0].getchildren()) == 2
    assert output[1] == (256, 256, 3)
    assert len(output[2].getchildren()) == 0


@e2e_pytest_unit
def test_convert_action_cls_dataset_to_datumaro(mocker) -> None:
    """Test convert_jester_dataset_to_datumaro function."""

    src_path = "dummy_src_path"
    ann_file = "dummy_ann_file"

    with tempfile.TemporaryDirectory() as dst_path:
        mocker.patch("otx.algorithms.action.utils.convert_public_data_to_cvat.open", return_value=MockFileObject())
        mocker.patch("otx.algorithms.action.utils.convert_public_data_to_cvat.pathlib.Path.mkdir", return_value=True)
        # mocker.patch("otx.algorithms.action.utils.convert_public_data_to_cvat.os.makedirs", return_value=True)
        mocker.patch("otx.algorithms.action.utils.convert_public_data_to_cvat.shutil.copy", return_value=True)
        mocker.patch(
            "otx.algorithms.action.utils.convert_public_data_to_cvat.generate_default_cvat_xml_fields",
            return_value=([], (256, 256, 3), []),
        )
        mocker.patch(
            "otx.algorithms.action.utils.convert_public_data_to_cvat.os.listdir", return_value=(["frame0", "frame1"])
        )
        mocker.patch(
            "otx.algorithms.action.utils.convert_public_data_to_cvat.etree.ElementTree", return_value=MockElementTree()
        )
        convert_action_cls_dataset_to_datumaro(src_path, dst_path, ann_file)


@e2e_pytest_unit
def test_convert_ava_dataset_to_datumaro(mocker) -> None:
    """Test convert_ava_dataset_to_datumaro function."""

    src_path = "dummy_src_path"
    ann_file = "dummy_ann_file"

    with tempfile.TemporaryDirectory() as dst_path:
        mocker.patch(
            "otx.algorithms.action.utils.convert_public_data_to_cvat.read_ava_csv",
            return_value={"video_0": {"frame_idx": [[0, 0, 1, 1, "action"]]}},
        )
        mocker.patch("otx.algorithms.action.utils.convert_public_data_to_cvat.os.listdir", return_value=["video_0"])
        mocker.patch("otx.algorithms.action.utils.convert_public_data_to_cvat.shutil.copytree", return_value=True)
        mocker.patch(
            "otx.algorithms.action.utils.convert_public_data_to_cvat.generate_default_cvat_xml_fields",
            return_value=([], (256, 256, 3), []),
        )
        mocker.patch(
            "otx.algorithms.action.utils.convert_public_data_to_cvat.etree.ElementTree", return_value=MockElementTree()
        )
        convert_ava_dataset_to_datumaro(src_path, dst_path, ann_file)


@e2e_pytest_unit
def test_rename_and_copy(mocker) -> None:
    mocker.patch("otx.algorithms.action.utils.convert_public_data_to_cvat.shutil.copy2", return_value=True)
    frame_name = "root/vid/frame_1.png"
    rename_and_copy(frame_name, frame_name)
    frame_name = "root/vid/1.png"
    rename_and_copy(frame_name, frame_name)


@e2e_pytest_unit
def test_read_ava_csv(mocker) -> None:
    mocker.patch("otx.algorithms.action.utils.convert_public_data_to_cvat.open", return_value=MockCsvFile())
    annot_info = read_ava_csv("dummy_path")
    assert len(annot_info) == 1
    assert len(annot_info["0"]) == 5
    assert annot_info["0"][0] == [["0", "0", "1", "1", "0"]]


@e2e_pytest_unit
@pytest.mark.parametrize("task", ["action_classification", "action_detection", "pose_estimation"])
def test_main(task, mocker) -> None:
    """Test main function."""
    mocker.patch(
        "otx.algorithms.action.utils.convert_public_data_to_cvat.parse_args", return_value=mocker.MagicMock(task=task)
    )
    mocker.patch(
        "otx.algorithms.action.utils.convert_public_data_to_cvat.convert_action_cls_dataset_to_datumaro",
        return_value=True,
    )
    mocker.patch(
        "otx.algorithms.action.utils.convert_public_data_to_cvat.convert_ava_dataset_to_datumaro", return_value=True
    )
    main()
