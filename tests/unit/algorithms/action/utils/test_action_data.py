"""Unit Test for otx.algorithms.action.utils.data."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np

from otx.algorithms.action.utils.data import (
    find_label_by_name,
    load_cls_annotations,
    load_cls_dataset,
    load_det_annotations,
    load_det_dataset,
)
from otx.api.entities.id import ID
from otx.api.entities.label import Domain, LabelEntity
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.action.test_helpers import generate_labels


@e2e_pytest_unit
def test_find_label_by_name() -> None:
    labels = generate_labels(3, Domain.ACTION_CLASSIFICATION)
    assert find_label_by_name(labels, "1", Domain.ACTION_CLASSIFICATION).name == "1"
    assert find_label_by_name(labels, "5", Domain.ACTION_CLASSIFICATION).name == "5"


@e2e_pytest_unit
def test_load_cls_annotations(mocker) -> None:
    class MockFile:
        def __init__(self, *args, **kwargs):
            self.lines = [
                "# Comment\n",
                "vid_0 5 1\n",
                "vid_1 5 1\n",
                "vid_2 5 1\n",
                "vid_3 5 2\n",
                "vid_4 5 2\n",
            ]

        def __enter__(self):
            return self.lines

        def __exit__(self, *args, **kwargs):
            return None

    mocker.patch("otx.algorithms.action.utils.data.open", return_value=MockFile())

    video_infos = load_cls_annotations("ann_file", "data_root")
    assert len(video_infos) == 5
    assert video_infos[0]["frame_dir"] == "data_root/vid_0"
    assert video_infos[0]["total_frames"] == 5
    assert video_infos[0]["label"] == 1


@e2e_pytest_unit
def test_load_det_annotations(mocker) -> None:
    class MockFile:
        def __init__(self, *args, **kwargs):
            self.lines = [
                "vid_0,0,0,0,1,1,1,0\n",
                "vid_0,1,0,0,1,1,1,1\n",
                "vid_0,1,0,0,1,1,2,1\n",
                "vid_0,2,0,0,1,1,1,2\n",
                "vid_0,3,0,0,1,1,2,3\n",
                "vid_0,4,0,0,1,1,2,4\n",
            ]

        def __enter__(self):
            return self.lines

        def __exit__(self, *args, **kwargs):
            return None

    mocker.patch("otx.algorithms.action.utils.data.open", return_value=MockFile())
    video_infos = load_det_annotations("ann_file", "data_root")
    assert len(video_infos) == 5
    assert video_infos[1]["frame_dir"] == "data_root/vid_0"
    assert video_infos[1]["video_id"] == "vid_0"
    assert video_infos[1]["timestamp"] == 1
    assert video_infos[1]["img_key"] == "vid_0,1"
    assert np.all(video_infos[1]["ann"]["gt_bboxes"] == np.array([[0.0, 0.0, 1.0, 1.0]]))
    assert np.all(video_infos[1]["ann"]["gt_labels"] == np.array([1, 2]))
    assert np.all(video_infos[1]["ann"]["entity_ids"] == np.array([1]))


@e2e_pytest_unit
def test_load_cls_dataset(mocker) -> None:
    """Test load_cls_dataset function."""

    mocker.patch(
        "otx.algorithms.action.utils.data.load_cls_annotations",
        return_value=[{"frame_dir": "data_root/vid_0", "total_frames": 5, "label": 1}],
    )
    mocker.patch(
        "otx.algorithms.action.utils.data.find_label_by_name",
        return_value=LabelEntity(name="1", domain=Domain.ACTION_CLASSIFICATION, id=ID(1)),
    )
    items = load_cls_dataset("ann_file", "data_root", Domain.ACTION_CLASSIFICATION)
    assert len(items) == 1
    assert items[0].media == {"frame_dir": "data_root/vid_0", "total_frames": 5}
    assert items[0].annotation_scene.get_labels()[0].name == "1"


@e2e_pytest_unit
def test_load_det_dataset(mocker) -> None:
    """Test load_det_dataset function."""

    mocker.patch(
        "otx.algorithms.action.utils.data.load_det_annotations",
        return_value=[
            {
                "frame_dir": "data_root/vid_0",
                "video_id": "vid_0",
                "timestamp": 0,
                "img_key": "vid_0,0",
                "ann": {
                    "gt_bboxes": np.array([[0.0, 0.0, 1.0, 1.0]]),
                    "gt_labels": [np.array([1])],
                    "entity_ids": np.array([0]),
                },
                "width": 320,
                "height": 240,
            }
        ],
    )
    mocker.patch(
        "otx.algorithms.action.utils.data.find_label_by_name",
        return_value=LabelEntity(name="1", domain=Domain.ACTION_DETECTION, id=ID(1)),
    )
    items = load_det_dataset("ann_file", "data_root", Domain.ACTION_DETECTION)
    assert len(items) == 1
    assert items[0].media == {
        "frame_dir": "data_root/vid_0",
        "video_id": "vid_0",
        "timestamp": 0,
        "img_key": "vid_0,0",
        "width": 320,
        "height": 240,
    }
    assert items[0].annotation_scene.get_labels()[0].name == "1"
