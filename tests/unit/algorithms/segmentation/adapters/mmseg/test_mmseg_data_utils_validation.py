# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import os
import os.path as osp
from tempfile import TemporaryDirectory

import cv2
import mmcv
import numpy as np
import pytest

from otx.algorithms.segmentation.adapters.mmseg.utils.data_utils import (
    add_labels,
    check_labels,
    create_annotation_from_hard_seg_map,
    create_pseudo_masks,
    load_dataset_items,
)
from otx.api.entities.label import Domain, LabelEntity
from tests.test_suite.e2e_test_system import e2e_pytest_unit


def label_entity(name="test label", id=None) -> LabelEntity:
    return LabelEntity(name=name, id=id, domain=Domain.SEGMENTATION)


def generate_random_single_image(filename: str, width: int = 10, height: int = 10) -> None:
    img: np.ndarray = np.uint8(np.random.random((height, width, 3)) * 255)
    cv2.imwrite(filename, img)


class TestMMSegDataUtilsValidation:
    @pytest.fixture(autouse=True)
    def setUp(self) -> None:

        self.hard_seg_map: np.ndarray = np.array([[1, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
        self.labels: list = [label_entity("class_0", id="00000000"), label_entity("class_1", id="00000001")]

    @e2e_pytest_unit
    def test_create_annotation_from_hard_seg_map(self) -> None:
        annotations: list = create_annotation_from_hard_seg_map(self.hard_seg_map, self.labels)

        assert len(annotations) == 1
        assert annotations[0].get_labels()[0].label.name == "class_0"

    @e2e_pytest_unit
    def test_check_labels(self) -> None:
        check_labels(cur_labels=self.labels, new_labels=[("class_1", None), ("class_0", None)])
        # function doesn't return anything, but throws exeption in case of failure
        assert True

    @e2e_pytest_unit
    def test_add_labels(self) -> None:
        add_labels(cur_labels=self.labels, new_labels=[("class_2", None)])

        assert len(self.labels) == 3

    @e2e_pytest_unit
    def test_create_pseudo_masks_fh_mode(self, mocker) -> None:
        mocker.patch("cv2.imread", return_value=np.array([[1, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]))
        mocker.patch("cv2.imwrite")
        mocker.patch("os.listdir", return_value=["image.jpg"])

        ann_file_path: str = "ann_file_path"
        data_root_dir: str = "data_root_dir"

        # ann_file_path dir should not exist
        create_pseudo_masks(ann_file_path, data_root_dir)

        assert osp.exists(osp.join(ann_file_path, "meta.json"))

        os.remove(f"{ann_file_path}/meta.json")
        os.rmdir(ann_file_path)

    @e2e_pytest_unit
    def test_load_dataset_items(self) -> None:

        tmp_dir: TemporaryDirectory = TemporaryDirectory()

        generate_random_single_image(osp.join(tmp_dir.name, "image.jpg"))
        generate_random_single_image(osp.join(tmp_dir.name, "image.png"))
        fake_json_file: str = osp.join(tmp_dir.name, "meta.json")
        mmcv.dump({"labels_map": []}, fake_json_file)

        # ann_file_path dir should exist
        dataset_items: list = load_dataset_items(tmp_dir.name, tmp_dir.name)

        assert len(dataset_items) == 1
