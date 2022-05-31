# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
from detection_tasks.extension.utils.pipelines import (
    LoadAnnotationFromOTEDataset,
    LoadImageFromOTEDataset,
)

from ote_sdk.test_suite.e2e_test_system import e2e_pytest_unit
from ote_sdk.tests.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)


class TestLoadImageFromOTEDatasetInputParamsValidation:
    @e2e_pytest_unit
    def test_load_image_from_ote_dataset_init_params_validation(self):
        """
        <b>Description:</b>
        Check LoadImageFromOTEDataset object initialization parameters validation

        <b>Input data:</b>
        "to_float32" non-bool parameter

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        LoadImageFromOTEDataset object initialization parameter
        """
        with pytest.raises(ValueError):
            LoadImageFromOTEDataset("unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_load_image_from_ote_dataset_call_params_validation(self):
        """
        <b>Description:</b>
        Check LoadImageFromOTEDataset object "__call__" method input parameters validation

        <b>Input data:</b>
        LoadImageFromOTEDataset object, "results" unexpected type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "__call__" method
        """
        load_image_from_ote_dataset = LoadImageFromOTEDataset()
        unexpected_int = 1
        for unexpected_value in [
            # Unexpected integer is specified as "results" parameter
            unexpected_int,
            # Unexpected integer is specified as "results" dictionary key
            {"result_1": "some results", unexpected_int: "unexpected results"},
        ]:
            with pytest.raises(ValueError):
                load_image_from_ote_dataset.__call__(results=unexpected_value)


class TestLoadAnnotationFromOTEDatasetInputParamsValidation:
    @e2e_pytest_unit
    def test_load_annotation_from_ote_dataset_init_params_validation(self):
        """
        <b>Description:</b>
        Check LoadAnnotationFromOTEDataset object initialization parameters validation

        <b>Input data:</b>
        LoadAnnotationFromOTEDataset object initialization parameters with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        LoadAnnotationFromOTEDataset object initialization parameter
        """
        correct_values_dict = {
            "min_size": 1,
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "min_size" parameter
            ("min_size", unexpected_str),
            # Unexpected string is specified as "with_bbox" parameter
            ("with_bbox", unexpected_str),
            # Unexpected string is specified as "with_label" parameter
            ("with_label", unexpected_str),
            # Unexpected string is specified as "with_mask" parameter
            ("with_mask", unexpected_str),
            # Unexpected string is specified as "with_seg" parameter
            ("with_seg", unexpected_str),
            # Unexpected string is specified as "poly2mask" parameter
            ("poly2mask", unexpected_str),
            # Unexpected string is specified as "with_text" parameter
            ("with_text", unexpected_str),
            # Unexpected string is specified as "domain" parameter
            ("domain", unexpected_str),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=LoadAnnotationFromOTEDataset,
        )

    @e2e_pytest_unit
    def test_load_annotation_from_ote_dataset_call_params_validation(self):
        """
        <b>Description:</b>
        Check LoadAnnotationFromOTEDataset object "__call__" method input parameters validation

        <b>Input data:</b>
        LoadAnnotationFromOTEDataset object, "results" unexpected type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "__call__" method
        """
        load_annotation_from_ote_dataset = LoadAnnotationFromOTEDataset(min_size=1)
        unexpected_int = 1
        for unexpected_value in [
            # Unexpected integer is specified as "results" parameter
            unexpected_int,
            # Unexpected integer is specified as "results" dictionary key
            {"result_1": "some results", unexpected_int: "unexpected results"},
        ]:
            with pytest.raises(ValueError):
                load_annotation_from_ote_dataset(results=unexpected_value)
