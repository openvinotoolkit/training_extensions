# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest

from otx.algorithms.common.utils import get_task_class, load_template
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestCommonUtilsInputParamsValidation:
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
