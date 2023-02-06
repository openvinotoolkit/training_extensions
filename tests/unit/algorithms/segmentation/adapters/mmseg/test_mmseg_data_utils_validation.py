# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import os
import os.path as osp
import tempfile
from os import makedirs, remove


import cv2
import mmcv
import numpy as np
import pytest

from otx.algorithms.segmentation.adapters.mmseg.utils.data_utils import (
    abs_path_if_valid,
    add_labels,
    check_labels,
    create_annotation_from_hard_seg_map,
    create_pseudo_masks,
    get_classes_from_annotation,
    get_extended_label_names,
    load_dataset_items,
    load_labels_from_annotation,
)
from otx.api.entities.id import ID
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.image import Image
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.api.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)

def label_entity(name="test label", id=None) -> LabelEntity:
    return LabelEntity(name=name, id=id, domain=Domain.SEGMENTATION)

def generate_random_single_image(filename: str, width: int = 10, height: int = 10) -> None:
    img = np.uint8(np.random.random((height, width, 3)) * 255)
    cv2.imwrite(filename, img)


class TestMMSegDataUtilsValidation:

    @pytest.fixture(autouse=True)
    def setUp(self) -> None:

        self.hard_seg_map = np.array([[1, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
        self.labels = [label_entity("class_0", id="00000000"), label_entity("class_1", id="00000001")]

    @e2e_pytest_unit
    def test_create_annotation_from_hard_seg_map(self):
        annotations = create_annotation_from_hard_seg_map(self.hard_seg_map, self.labels)

        assert len(annotations) == 1
        assert annotations[0].get_labels()[0].label.name == "class_0"

    @e2e_pytest_unit
    def test_check_labels(self):
        check_labels(cur_labels = self.labels, new_labels = [("class_1", None), ("class_0", None)])
    

    @e2e_pytest_unit
    def test_add_labels(self):
        add_labels(cur_labels = self.labels, new_labels = [("class_2", None)])

        assert len(self.labels) == 3

    @e2e_pytest_unit
    def test_create_pseudo_masks_fh_mode(self, mocker):
        mocker.patch("cv2.imread", return_value = np.array([[1, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]))
        mocker.patch("cv2.imwrite")
        mocker.patch("os.listdir", return_value = ['image.jpg'])

        ann_file_path = "ann_file_path"
        data_root_dir = "data_root_dir"

        # remove(f'{ann_file_path}/meta.json')
        # os.rmdir(ann_file_path)

        # ann_file_path dir should not exist
        create_pseudo_masks(ann_file_path, data_root_dir)
        # load_labels_from_annotation(ann_file_path)

        # assert osp.isdir(ann_file_path)
        assert osp.exists(osp.join(ann_file_path, "meta.json"))

        remove(f'{ann_file_path}/meta.json')
        os.rmdir(ann_file_path)
        # remove(f'{data_root_dir}/image.jpg')
        # os.rmdir(data_root_dir)

    @e2e_pytest_unit
    def test_load_dataset_items(self, mocker):

        tmp_dir = tempfile.TemporaryDirectory()

        generate_random_single_image(osp.join(tmp_dir.name, "image.jpg"))
        generate_random_single_image(osp.join(tmp_dir.name, "image.png"))
        fake_json_file = osp.join(tmp_dir.name, "meta.json")
        mmcv.dump({'labels_map': []}, fake_json_file)

        # ann_file_path dir should exist
        dataset_items = load_dataset_items(tmp_dir.name, tmp_dir.name)

        assert len(dataset_items) == 1
