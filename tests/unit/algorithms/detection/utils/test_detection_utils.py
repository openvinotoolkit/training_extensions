# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest

from otx.algorithms.detection.utils.utils import (
    ColorPalette,
    generate_label_schema,
    get_det_model_api_configuration,
)
from otx.api.entities.model_template import TaskType, task_type_to_label_domain
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.api.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)


@e2e_pytest_unit
def test_get_det_model_api_configuration():
    classes = ("rectangle", "ellipse", "triangle")
    label_schema = generate_label_schema(classes, task_type_to_label_domain(TaskType.DETECTION))
    det_thr = 0.5
    model_api_cfg = get_det_model_api_configuration(label_schema, TaskType.DETECTION, det_thr)

    assert len(model_api_cfg) > 0
    assert model_api_cfg[("model_info", "confidence_threshold")] == str(det_thr)


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


class TestOTXUtilsFunctionsInputParamsValidation:
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
