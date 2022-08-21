# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest

from ote_sdk.test_suite.e2e_test_system import e2e_pytest_unit
from segmentation_tasks.apis.segmentation.ote_utils import (
    get_activation_map,
    get_task_class,
    load_template,
)


class TestOTEUtilsFunctionsInputParamsValidation:
    @e2e_pytest_unit
    def test_load_template_params_validation(self):
        """
        <b>Description:</b>
        Check "load_template" function input parameters validation

        <b>Input data:</b>
        "path" unexpected file path string

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
                load_template(path=incorrect_parameter)

    @e2e_pytest_unit
    def test_get_task_class_input_params_validation(self):
        """
        <b>Description:</b>
        Check "get_task_class" function input parameters validation

        <b>Input data:</b>
        "path" unexpected file path string

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "get_task_class" function
        """
        with pytest.raises(ValueError):
            get_task_class(path=1)  # type: ignore

    @e2e_pytest_unit
    def test_get_activation_map_input_params_validation(self):
        """
        <b>Description:</b>
        Check "get_activation_map" function input parameters validation

        <b>Input data:</b>
        "features" unexpected type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "get_activation_map" function
        """
        with pytest.raises(ValueError):
            get_activation_map(features=None)  # type: ignore
