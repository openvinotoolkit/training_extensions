# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import os.path as osp
import tempfile
from os import remove

import mmcv
import numpy as np
import pytest

from otx.algorithms.segmentation.adapters.mmseg.utils.data_utils import (
    abs_path_if_valid,
    add_labels,
    check_labels,
    create_annotation_from_hard_seg_map,
    get_classes_from_annotation,
    get_extended_label_names,
    load_dataset_items,
    load_labels_from_annotation,
)
from otx.api.entities.label import Domain, LabelEntity
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.api.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)


def _create_dummy_coco_json(json_name):
    image = {
        "id": 0,
        "width": 640,
        "height": 640,
        "file_name": "fake_name.jpg",
    }

    annotation_1 = {
        "id": 1,
        "image_id": 0,
        "category_id": 0,
        "area": 400,
        "bbox": [50, 60, 20, 20],
        "iscrowd": 0,
    }

    annotation_2 = {
        "id": 2,
        "image_id": 0,
        "category_id": 0,
        "area": 900,
        "bbox": [100, 120, 30, 30],
        "iscrowd": 0,
    }

    categories = [
        {
            "id": 0,
            "name": "car",
            "supercategory": "car",
        }
    ]

    fake_json = {
        "images": [image],
        "annotations": [annotation_1, annotation_2],
        "categories": categories,
    }

    mmcv.dump(fake_json, json_name)


class TestMMSegDataUtilsInputParamsValidation:
    @e2e_pytest_unit
    def test_abs_path_if_valid_input_params_validation(self):
        """
        <b>Description:</b>
        Check "abs_path_if_valid" function input parameters validation

        <b>Input data:</b>
        "value" unexpected object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as input parameter for
        "abs_path_if_valid" function
        """
        for unexpected_value in [
            # Unexpected integer is specified as "value" parameter
            1,
            # Empty string is specified as "value" parameter
            "",
            # Path with null character is specified as "value" parameter
            "./\0null_char",
            # Path with non-printable character is specified as "value" parameter
            "./\non_printable_char",
        ]:
            with pytest.raises(ValueError):
                abs_path_if_valid(value=unexpected_value)

    @e2e_pytest_unit
    def test_get_classes_from_annotation_input_params_validation(self):
        """
        <b>Description:</b>
        Check "get_classes_from_annotation" function input parameters validation

        <b>Input data:</b>
        "annot_path" unexpected object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as input parameter for
        "get_classes_from_annotation" function
        """
        for unexpected_value in [
            # non string object is specified as "annot_path" parameter
            1,
            # Empty string is specified as "annot_path" parameter
            "",
            # Path to file with unexpected extension is specified as "annot_path" parameter
            "./unexpected_extension.yaml",
            # Path to non-existing file is specified as "annot_path" parameter
            "./non_existing.json",
            # Path with null character is specified as "annot_path" parameter
            "./null\0char.json",
            # Path with non-printable character is specified as "annot_path" parameter
            "./\non_printable_char.json",
        ]:
            with pytest.raises(ValueError):
                get_classes_from_annotation(annot_path=unexpected_value)

    @e2e_pytest_unit
    def test_create_annotation_from_hard_seg_map_input_params_validation(self):
        """
        <b>Description:</b>
        Check "create_annotation_from_hard_seg_map" function input parameters validation

        <b>Input data:</b>
        "create_annotation_from_hard_seg_map" function unexpected-type input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "create_annotation_from_hard_seg_map" function
        """
        label = LabelEntity(name="test label", domain=Domain.SEGMENTATION)
        correct_values_dict = {
            "hard_seg_map": np.random.rand(2, 2),
            "labels": [label],
        }
        unexpected_int = 1
        unexpected_values = [
            # Unexpected integer is specified as "dataset_item" parameter
            ("hard_seg_map", unexpected_int),
            # Unexpected integer is specified as "labels" parameter
            ("labels", unexpected_int),
            # Unexpected integer is specified as nested label
            ("labels", [label, unexpected_int]),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=create_annotation_from_hard_seg_map,
        )

    @e2e_pytest_unit
    def test_load_labels_from_annotation_input_params_validation(self):
        """
        <b>Description:</b>
        Check "load_labels_from_annotation" function input parameters validation

        <b>Input data:</b>
        "ann_dir" unexpected object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as input parameter for
        "load_labels_from_annotation" function
        """
        for unexpected_value in [
            # Unexpected integer is specified as "ann_dir" parameter
            1,
            # Empty string is specified as "ann_dir" parameter
            "",
            # Path with null character is specified as "ann_dir" parameter
            "./\0null_char",
            # Path with non-printable character is specified as "ann_dir" parameter
            "./\non_printable_char",
        ]:
            with pytest.raises(ValueError):
                load_labels_from_annotation(ann_dir=unexpected_value)

    @e2e_pytest_unit
    def test_add_labels_input_params_validation(self):
        """
        <b>Description:</b>
        Check "add_labels" function input parameters validation

        <b>Input data:</b>
        "add_labels" function unexpected-type input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "add_labels" function
        """
        label = LabelEntity(name="test label", domain=Domain.SEGMENTATION)
        correct_values_dict = {
            "cur_labels": [label],
            "new_labels": [("label_name1", "label_id1")],
        }
        unexpected_int = 1
        unexpected_values = [
            # Unexpected integer is specified as "cur_labels" parameter
            ("cur_labels", unexpected_int),
            # Unexpected integer is specified as nested label
            ("cur_labels", [label, unexpected_int]),
            # Unexpected integer is specified as "new_labels" parameter
            ("new_labels", unexpected_int),
            # Unexpected integer is specified as nested new_label
            ("new_labels", [("label_name1", "label_id1"), unexpected_int]),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=add_labels,
        )

    @e2e_pytest_unit
    def test_check_labels_input_params_validation(self):
        """
        <b>Description:</b>
        Check "check_labels" function input parameters validation

        <b>Input data:</b>
        "check_labels" function unexpected-type input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "check_labels" function
        """
        label = LabelEntity(name="test label", domain=Domain.SEGMENTATION)
        correct_values_dict = {
            "cur_labels": [label],
            "new_labels": [("label_name1", "label_id1")],
        }
        unexpected_int = 1
        unexpected_values = [
            # Unexpected integer is specified as "cur_labels" parameter
            ("cur_labels", unexpected_int),
            # Unexpected integer is specified as nested label
            ("cur_labels", [label, unexpected_int]),
            # Unexpected integer is specified as "new_labels" parameter
            ("new_labels", unexpected_int),
            # Unexpected integer is specified as nested new_label
            ("new_labels", [("label_name1", "label_id1"), unexpected_int]),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=check_labels,
        )

    @e2e_pytest_unit
    def test_get_extended_label_names_input_params_validation(self):
        """
        <b>Description:</b>
        Check "get_extended_label_names" function input parameters validation

        <b>Input data:</b>
        "labels" unexpected object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as input parameter for
        "get_extended_label_names" function
        """
        label = LabelEntity(name="test label", domain=Domain.SEGMENTATION)
        unexpected_int = 1
        for unexpected_value in [
            # Unexpected integer is specified as "labels" parameter
            unexpected_int,
            # Empty string is specified as nested label
            [label, unexpected_int],
        ]:
            with pytest.raises(ValueError):
                get_extended_label_names(labels=unexpected_value)

    @e2e_pytest_unit
    def test_load_dataset_items_params_validation(self):
        """
        <b>Description:</b>
        Check "load_dataset_items" function input parameters validation

        <b>Input data:</b>
        "load_dataset_items" function unexpected-type input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "load_dataset_items" function
        """
        tmp_dir = tempfile.TemporaryDirectory()
        fake_json_file = osp.join(tmp_dir.name, "fake_data.json")
        _create_dummy_coco_json(fake_json_file)
        label = LabelEntity(name="test label", domain=Domain.DETECTION)
        correct_values_dict = {
            "ann_file_path": fake_json_file,
            "data_root_dir": tmp_dir.name,
        }
        unexpected_int = 1
        unexpected_values = [
            # Unexpected integer is specified as "ann_file_path" parameter
            ("ann_file_path", unexpected_int),
            # Empty string is specified as "ann_file_path" parameter
            ("ann_file_path", ""),
            # Path with null character is specified as "ann_file_path" parameter
            ("ann_file_path", osp.join(tmp_dir.name, "\0fake_data.json")),
            # Path with non-printable character is specified as "ann_file_path" parameter
            ("ann_file_path", osp.join(tmp_dir.name, "\nfake_data.json")),
            # Unexpected integer is specified as "data_root_dir" parameter
            ("data_root_dir", unexpected_int),
            # Empty string is specified as "data_root_dir" parameter
            ("data_root_dir", ""),
            # Path with null character is specified as "data_root_dir" parameter
            ("data_root_dir", "./\0null_char"),
            # Path with non-printable character is specified as "data_root_dir" parameter
            ("data_root_dir", "./\non_printable_char"),
            # Unexpected integer is specified as "subset" parameter
            ("subset", unexpected_int),
            # Unexpected integer is specified as "labels_list" parameter
            ("labels_list", unexpected_int),
            # Unexpected integer is specified as nested label
            ("labels_list", [label, unexpected_int]),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=load_dataset_items,
        )
        remove(fake_json_file)
