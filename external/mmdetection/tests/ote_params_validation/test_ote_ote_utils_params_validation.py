# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
from detection_tasks.apis.detection.ote_utils import (
    ColorPalette,
    generate_label_schema,
    get_task_class,
    load_template,
)

from ote_sdk.test_suite.e2e_test_system import e2e_pytest_unit
from ote_sdk.tests.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)


class TestColorPaletteInputParamsValidation:
    @staticmethod
    def color_palette():
        return ColorPalette(1)

    @e2e_pytest_unit
    def test_color_palette_init_params_validation(self):
        """
        <b>Description:</b>
        Check ColorPalette object initialization parameters validation

        <b>Input data:</b>
        ColorPalette object initialization parameters with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        ColorPalette object initialization parameter
        """
        correct_values_dict = {
            "n": 1,
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "n" parameter
            ("n", unexpected_str),
            # Unexpected string is specified as "rng" parameter
            ("rng", unexpected_str),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=ColorPalette,
        )

    @e2e_pytest_unit
    def test_color_palette_get_item_params_validation(self):
        """
        <b>Description:</b>
        Check ColorPalette object "__getitem__" method input parameters validation

        <b>Input data:</b>
        ColorPalette object, "n" non-integer object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "__getitem__" method
        """
        color_palette = self.color_palette()
        with pytest.raises(ValueError):
            color_palette.__getitem__("unexpected string")  # type: ignore


class TestOTEUtilsFunctionsInputParamsValidation:
    @e2e_pytest_unit
    def test_generate_label_schema_input_params_validation(self):
        """
        <b>Description:</b>
        Check "generate_label_schema" function input parameters validation

        <b>Input data:</b>
        "generate_label_schema" function unexpected-type input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "generate_label_schema" function
        """
        correct_values_dict = {
            "label_names": ["label_1", "label_2"],
        }
        unexpected_int = 1
        unexpected_values = [
            # Unexpected integer is specified as "label_names" parameter
            ("label_names", unexpected_int),
            # Unexpected integer is specified as nested label name
            ("label_names", ["label_1", unexpected_int]),
            # Unexpected integer is specified as "label_domain" parameter
            ("label_domain", unexpected_int),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=generate_label_schema,
        )

    @e2e_pytest_unit
    def test_load_template_params_validation(self):
        """
        <b>Description:</b>
        Check "load_template" function input parameters validation

        <b>Input data:</b>
        "path" unexpected string with yaml file object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "load_template" function
        """
        for incorrect_parameter in [
            # Unexpected integer is specified as "path" parameter
            1,
            # Empty string is specified as "path" parameter
            "",
            # Path to non-existing file is specified as "path" parameter
            "./non_existing.yaml",
            # Path to non-yaml file is specified as "path" parameter
            "./unexpected_type.jpg",
            # Path Null character is specified in "path" parameter
            "./null\0char.yaml",
            # Path with non-printable character is specified as "path" parameter
            "./non\nprintable.yaml",
        ]:
            with pytest.raises(ValueError):
                load_template(incorrect_parameter)

    @e2e_pytest_unit
    def test_get_task_class_input_params_validation(self):
        """
        <b>Description:</b>
        Check "get_task_class" function input parameters validation

        <b>Input data:</b>
        "path" non string-type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "get_task_class" function
        """
        with pytest.raises(ValueError):
            get_task_class(path=1)  # type: ignore
