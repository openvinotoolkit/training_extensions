# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import os.path as osp
import tempfile

import mmcv
import pytest

from otx.algorithms.detection.utils.data import (
    CocoDataset,
    LoadAnnotations,
    find_label_by_name,
    format_list_to_str,
    get_anchor_boxes,
    get_classes_from_annotation,
    get_sizes_from_dataset_entity,
    load_dataset_items_coco_format,
)
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label import Domain, LabelEntity
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.api.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)

# TODO: Need to add adaptive_tile_params unit-test


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


class TestDataUtilsFunctionsInputParamsValidation:
    @e2e_pytest_unit
    def test_get_classes_from_annotation_input_params_validation(self):
        """
        <b>Description:</b>
        Check "get_classes_from_annotation" function input parameters validation

        <b>Input data:</b>
        "path" unexpected object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as input parameter for
        "get_classes_from_annotation" function
        """
        for unexpected_value in [
            # non string object is specified as "path" parameter
            1,
            # Empty string is specified as "path" parameter
            "",
            # Path to file with unexpected extension is specified as "path" parameter
            "./unexpected_extension.yaml",
            # Path to non-existing file is specified as "path" parameter
            "./non_existing.json",
            # Path with null character is specified as "path" parameter
            "./null\0char.json",
            # Path with non-printable character is specified as "path" parameter
            "./\non_printable_char.json",
        ]:
            with pytest.raises(ValueError):
                get_classes_from_annotation(path=unexpected_value)

    @e2e_pytest_unit
    def test_find_label_by_name_params_validation(self):
        """
        <b>Description:</b>
        Check "find_label_by_name" function input parameters validation

        <b>Input data:</b>
        "find_label_by_name" function unexpected-type input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "find_label_by_name" function
        """
        label = LabelEntity(name="test label", domain=Domain.DETECTION)
        correct_values_dict = {
            "labels": [label],
            "name": "test label",
            "domain": Domain.DETECTION,
        }
        unexpected_int = 1
        unexpected_values = [
            # Unexpected integer is specified as "labels" parameter
            ("labels", unexpected_int),
            # Unexpected integer is specified as nested label
            ("labels", [label, unexpected_int]),
            # Unexpected integer is specified as "name" parameter
            ("name", unexpected_int),
            # Unexpected integer is specified as "domain" parameter
            ("domain", unexpected_int),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=find_label_by_name,
        )

    @e2e_pytest_unit
    def test_load_dataset_items_coco_format_params_validation(self):
        """
        <b>Description:</b>
        Check "load_dataset_items_coco_format" function input parameters validation

        <b>Input data:</b>
        "load_dataset_items_coco_format" function unexpected-type input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "load_dataset_items_coco_format" function
        """
        tmp_dir = tempfile.TemporaryDirectory()
        fake_json_file = osp.join(tmp_dir.name, "fake_data.json")
        _create_dummy_coco_json(fake_json_file)

        label = LabelEntity(name="test label", domain=Domain.DETECTION)
        correct_values_dict = {
            "ann_file_path": fake_json_file,
            "data_root_dir": tmp_dir.name,
            "domain": Domain.DETECTION,
        }
        unexpected_int = 1
        unexpected_values = [
            # Unexpected integer is specified as "ann_file_path" parameter
            ("ann_file_path", unexpected_int),
            # Empty string is specified as "ann_file_path" parameter
            ("ann_file_path", ""),
            # Path to non-json file is specified as "ann_file_path" parameter
            ("ann_file_path", osp.join(tmp_dir.name, "non_json.jpg")),
            # Path with null character is specified as "ann_file_path" parameter
            ("ann_file_path", osp.join(tmp_dir.name, "\0fake_data.json")),
            # Path with non-printable character is specified as "ann_file_path" parameter
            ("ann_file_path", osp.join(tmp_dir.name, "\nfake_data.json")),
            # Path to non-existing file is specified as "ann_file_path" parameter
            ("ann_file_path", osp.join(tmp_dir.name, "non_existing.json")),
            # Unexpected integer is specified as "data_root_dir" parameter
            ("data_root_dir", unexpected_int),
            # Empty string is specified as "data_root_dir" parameter
            ("data_root_dir", ""),
            # Path with null character is specified as "data_root_dir" parameter
            ("data_root_dir", "./\0null_char"),
            # Path with non-printable character is specified as "data_root_dir" parameter
            ("data_root_dir", "./\non_printable_char"),
            # Unexpected integer is specified as "domain" parameter
            ("domain", unexpected_int),
            # Unexpected integer is specified as "subset" parameter
            ("subset", unexpected_int),
            # Unexpected integer is specified as "labels_list" parameter
            ("labels_list", unexpected_int),
            # Unexpected integer is specified as nested label
            ("labels_list", [label, unexpected_int]),
            # Unexpected string is specified as "with_mask" parameter
            ("with_mask", "unexpected string"),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=load_dataset_items_coco_format,
        )

    @e2e_pytest_unit
    def test_get_sizes_from_dataset_entity_params_validation(self):
        """
        <b>Description:</b>
        Check "get_sizes_from_dataset_entity" function input parameters validation

        <b>Input data:</b>
        "get_sizes_from_dataset_entity" function unexpected-type input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "get_sizes_from_dataset_entity" function
        """
        correct_values_dict = {
            "dataset": DatasetEntity(),
            "target_wh": [(0.1, 0.1)],
        }
        unexpected_int = 1
        unexpected_values = [
            # Unexpected integer is specified as "dataset" parameter
            ("dataset", unexpected_int),
            # Unexpected integer is specified as "target_wh" parameter
            ("target_wh", unexpected_int),
            # Unexpected integer is specified as nested target_wh
            ("target_wh", [(0.1, 0.1), unexpected_int]),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=get_sizes_from_dataset_entity,
        )

    @e2e_pytest_unit
    def test_format_list_to_str_params_validation(self):
        """
        <b>Description:</b>
        Check "format_list_to_str" function input parameters validation

        <b>Input data:</b>
        "value_lists" unexpected type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "format_list_to_str" function
        """
        with pytest.raises(ValueError):
            format_list_to_str(value_lists="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_get_anchor_boxes_params_validation(self):
        """
        <b>Description:</b>
        Check "get_anchor_boxes" function input parameters validation

        <b>Input data:</b>
        "get_anchor_boxes" function unexpected-type input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "get_anchor_boxes" function
        """
        correct_values_dict = {
            "wh_stats": [("wh_stat_1", 1), ("wh_stat_2", 2)],
            "group_as": [0, 1, 2],
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "wh_stats" parameter
            ("wh_stats", unexpected_str),
            # Unexpected string is specified as nested "wh_stat"
            ("wh_stats", [("wh_stat_1", 1), unexpected_str]),
            # Unexpected string is specified as "group_as" parameter
            ("group_as", unexpected_str),
            # Unexpected string is specified as nested "group_as"
            ("group_as", [0, 1, 2, unexpected_str]),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=get_anchor_boxes,
        )


class TestLoadAnnotationsInputParamsValidation:
    @e2e_pytest_unit
    def test_load_annotations_init_params_validation(self):
        """
        <b>Description:</b>
        Check LoadAnnotations object initialization parameters validation

        <b>Input data:</b>
        LoadAnnotations object initialization parameters with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        LoadAnnotations initialization parameter
        """
        for parameter in ["with_bbox", "with_label", "with_mask"]:
            with pytest.raises(ValueError):
                LoadAnnotations(**{parameter: "unexpected string"})

    @e2e_pytest_unit
    def test_load_annotations_call_params_validation(self):
        """
        <b>Description:</b>
        Check LoadAnnotations object "__call__" method input parameters validation

        <b>Input data:</b>
        "results" parameter with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "__call__" method
        """
        load_annotations = LoadAnnotations()
        unexpected_int = 1
        for unexpected_value in [
            # Unexpected integer is specified as "results" parameter
            unexpected_int,
            # Unexpected integer is specified as "results" dictionary key
            {"result_1": "some results", unexpected_int: "unexpected results"},
        ]:
            with pytest.raises(ValueError):
                load_annotations(results=unexpected_value)


class TestCocoDatasetInputParamsValidation:
    @staticmethod
    def create_fake_json_file():
        tmp_dir = tempfile.TemporaryDirectory()
        fake_json_file = osp.join(tmp_dir.name, "fake_data.json")
        _create_dummy_coco_json(fake_json_file)
        return fake_json_file

    @staticmethod
    def dataset():
        tmp_dir = tempfile.TemporaryDirectory()
        fake_json_file = osp.join(tmp_dir.name, "fake_data.json")
        _create_dummy_coco_json(fake_json_file)
        return CocoDataset(fake_json_file)

    @e2e_pytest_unit
    def test_coco_dataset_init_params_validation(self):
        """
        <b>Description:</b>
        Check CocoDataset object initialization parameters validation

        <b>Input data:</b>
        CocoDataset object initialization parameters with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        CocoDataset object initialization parameter
        """
        tmp_dir = tempfile.TemporaryDirectory()
        fake_json_file = osp.join(tmp_dir.name, "fake_data.json")
        _create_dummy_coco_json(fake_json_file)

        correct_values_dict = {
            "ann_file": fake_json_file,
        }
        unexpected_str = "unexpected string"
        unexpected_int = 1
        unexpected_values = [
            # Unexpected integer is specified as "ann_file" parameter
            ("ann_file", unexpected_int),
            # Empty string is specified as "ann_file" parameter
            ("ann_file", ""),
            # Path to non-json file is specified as "ann_file" parameter
            ("ann_file", osp.join(tmp_dir.name, "non_json.jpg")),
            # Path with null character is specified as "ann_file" parameter
            ("ann_file", osp.join(tmp_dir.name, "\0fake_data.json")),
            # Path with non-printable character is specified as "ann_file" parameter
            ("ann_file", osp.join(tmp_dir.name, "\nfake_data.json")),
            # Path to non-existing file is specified as "ann_file" parameter
            ("ann_file", osp.join(tmp_dir.name, "non_existing.json")),
            # Unexpected integer is specified as "classes" parameter
            ("classes", unexpected_int),
            # Unexpected integer is specified nested class
            ("classes", ["class_1", unexpected_int]),
            # Unexpected integer is specified as "data_root" parameter
            ("data_root", unexpected_int),
            # Empty string is specified as "data_root" parameter
            ("data_root", ""),
            # Path with null character is specified as "data_root" parameter
            ("data_root", "./\0null_char"),
            # Path with non-printable character is specified as "data_root" parameter
            ("data_root", "./\non_printable_char"),
            # Unexpected integer is specified as "img_prefix" parameter
            ("img_prefix", unexpected_int),
            # Unexpected string is specified as "test_mode" parameter
            ("test_mode", unexpected_str),
            # Unexpected string is specified as "filter_empty_gt" parameter
            ("filter_empty_gt", unexpected_str),
            # Unexpected string is specified as "min_size" parameter
            ("min_size", unexpected_str),
            # Unexpected string is specified as "with_mask" parameter
            ("with_mask", unexpected_str),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=CocoDataset,
        )

    @e2e_pytest_unit
    def test_coco_dataset_pre_pipeline_params_validation(self):
        """
        <b>Description:</b>
        Check CocoDataset object "pre_pipeline" method input parameters validation

        <b>Input data:</b>
        CocoDataset object, "results" parameter with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "pre_pipeline" method
        """
        dataset = self.dataset()
        unexpected_int = 1
        for unexpected_value in [
            # Unexpected integer is specified as "results" parameter
            unexpected_int,
            # Unexpected integer is specified as "results" dictionary key
            {"result_1": "some results", unexpected_int: "unexpected results"},
        ]:
            with pytest.raises(ValueError):
                dataset.pre_pipeline(results=unexpected_value)

    @e2e_pytest_unit
    def test_coco_dataset_get_item_params_validation(self):
        """
        <b>Description:</b>
        Check CocoDataset object "__getitem__" method input parameters validation

        <b>Input data:</b>
        CocoDataset object, "idx" non-integer type parameter

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "__getitem__" method
        """
        dataset = self.dataset()
        with pytest.raises(ValueError):
            dataset.__getitem__(idx="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_coco_dataset_prepare_img_params_validation(self):
        """
        <b>Description:</b>
        Check CocoDataset object "prepare_img" method input parameters validation

        <b>Input data:</b>
        CocoDataset object, "idx" non-integer type parameter

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "prepare_img" method
        """
        dataset = self.dataset()
        with pytest.raises(ValueError):
            dataset.prepare_img(idx="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_coco_dataset_get_classes_params_validation(self):
        """
        <b>Description:</b>
        Check CocoDataset object "get_classes" method input parameters validation

        <b>Input data:</b>
        CocoDataset object, "classes" parameter with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "get_classes" method
        """
        dataset = self.dataset()
        unexpected_int = 1
        for unexpected_value in [
            # Unexpected integer is specified as "classes" parameter
            unexpected_int,
            # Unexpected integer is specified as nested "classes" element
            ["class_1", unexpected_int],
        ]:
            with pytest.raises(ValueError):
                dataset.get_classes(classes=unexpected_value)  # type: ignore

    @e2e_pytest_unit
    def test_coco_dataset_load_annotations_params_validation(self):
        """
        <b>Description:</b>
        Check CocoDataset object "load_annotations" method input parameters validation

        <b>Input data:</b>
        CocoDataset object, "ann_file" unexpected object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "load_annotations" method
        """
        dataset = self.dataset()
        for unexpected_value in [
            # Unexpected integer is specified as "ann_file" parameter
            1,
            # Empty string is specified as "ann_file" parameter
            "",
            # Path to non-existing file is specified as "ann_file" parameter
            "./non_existing.json",
            # Path to non-json file is specified as "ann_file" parameter
            "./unexpected_type.jpg",
            # Path Null character is specified in "ann_file" parameter
            "./null\0char.json",
            # Path with non-printable character is specified as "input_config" parameter
            "./null\nchar.json",
        ]:
            with pytest.raises(ValueError):
                dataset.load_annotations(ann_file=unexpected_value)

    @e2e_pytest_unit
    def test_coco_dataset_get_ann_info_params_validation(self):
        """
        <b>Description:</b>
        Check CocoDataset object "get_ann_info" method input parameters validation

        <b>Input data:</b>
        CocoDataset object, "idx" non-integer type parameter

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "get_ann_info" method
        """
        dataset = self.dataset()
        with pytest.raises(ValueError):
            dataset.get_ann_info(idx="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_coco_dataset_get_cat_ids_params_validation(self):
        """
        <b>Description:</b>
        Check CocoDataset object "get_cat_ids" method input parameters validation

        <b>Input data:</b>
        CocoDataset object, "idx" non-integer type parameter

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "get_cat_ids" method
        """
        dataset = self.dataset()
        with pytest.raises(ValueError):
            dataset.get_cat_ids(idx="unexpected string")  # type: ignore
