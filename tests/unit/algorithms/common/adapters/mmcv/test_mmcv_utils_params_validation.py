# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
from mmcv import Config

from otx.algorithms.common.adapters.mmcv.utils import (
    config_from_string,
    get_data_cfg,
    is_epoch_based_runner,
    patch_color_conversion,
    prepare_for_testing,
    prepare_work_dir,
    remove_from_config,
)
from otx.api.entities.datasets import DatasetEntity
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.api.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)

# TODO: Need to add {patch_data_pipeline, get_meta_keys} unit-test


class TestMMCVUtilsInputParamsValidation:
    @e2e_pytest_unit
    def test_is_epoch_based_runner_input_params_validation(self):
        """
        <b>Description:</b>
        Check "is_epoch_based_runner" function input parameters validation

        <b>Input data:</b>
        "runner_config" non-ConfigDict object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as input parameter for
        "runner_config" function
        """
        with pytest.raises(ValueError):
            is_epoch_based_runner(runner_config="unexpected_str")  # type: ignore

    @e2e_pytest_unit
    def test_prepare_for_testing_input_params_validation(self):
        """
        <b>Description:</b>
        Check "prepare_for_testing" function input parameters validation

        <b>Input data:</b>
        "prepare_for_testing" function input parameters with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as input parameter for
        "prepare_for_testing" function
        """
        correct_values_dict = {"config": Config(), "dataset": DatasetEntity()}
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "config" parameter
            ("config", unexpected_str),
            # Unexpected string is specified as "dataset" parameter
            ("dataset", unexpected_str),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=prepare_for_testing,
        )

    @e2e_pytest_unit
    def test_config_from_string_input_params_validation(self):
        """
        <b>Description:</b>
        Check "config_from_string" function input parameters validation

        <b>Input data:</b>
        "config_string" non-string type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as input parameter for
        "config_from_string" function
        """
        with pytest.raises(ValueError):
            config_from_string(config_string=1)  # type: ignore

    @e2e_pytest_unit
    def test_prepare_work_dir_input_params_validation(self):
        """
        <b>Description:</b>
        Check "prepare_work_dir" function input parameters validation

        <b>Input data:</b>
        "config" non-Config type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as input parameter for
        "prepare_work_dir" function
        """
        with pytest.raises(ValueError):
            prepare_work_dir(config=1)  # type: ignore

    @e2e_pytest_unit
    def test_remove_from_config_input_params_validation(self):
        """
        <b>Description:</b>
        Check "remove_from_config" function input parameters validation

        <b>Input data:</b>
        "remove_from_config" function input parameters with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as input parameter for
        "remove_from_config" function
        """
        correct_values_dict = {"config": Config(), "key": "key_1"}
        unexpected_int = 1
        unexpected_values = [
            # Unexpected integer is specified as "config" parameter
            ("config", unexpected_int),
            # Unexpected integer is specified as "key" parameter
            ("key", unexpected_int),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=remove_from_config,
        )

    @e2e_pytest_unit
    def test_get_data_cfg_input_params_validation(self):
        """
        <b>Description:</b>
        Check "get_data_cfg" function input parameters validation

        <b>Input data:</b>
        "get_data_cfg" function input parameters with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as input parameter for
        "get_data_cfg" function
        """
        correct_values_dict = {
            "config": Config(),
        }
        unexpected_int = 1
        unexpected_values = [
            # Unexpected integer is specified as "config" parameter
            ("config", unexpected_int),
            # Unexpected integer is specified as "subset" parameter
            ("subset", unexpected_int),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=get_data_cfg,
        )

    @e2e_pytest_unit
    def test_patch_data_pipeline_input_params_validation(self):
        """
        <b>Description:</b>
        Check "patch_data_pipeline" function input parameters validation

        <b>Input data:</b>
        "patch_data_pipeline" function input parameters with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as input parameter for
        "patch_data_pipeline" function
        """
        correct_values_dict = {
            "config": Config(),
            "data_pipeline": "otx/algorithms/classification/configs/base/data/data_pipeline.py",
        }
        unexpected_int = 1
        unexpected_values = [
            # Unexpected integer is specified as "config" parameter
            ("config", unexpected_int),
            # Unexpected integer is specified as "subset" parameter
            ("subset", unexpected_int),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=get_data_cfg,
        )

    @e2e_pytest_unit
    def test_patch_color_conversion_input_params_validation(self):
        """
        <b>Description:</b>
        Check "patch_color_conversion" function input parameters validation

        <b>Input data:</b>
        "pipeline" unexpected type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as input parameter for
        "patch_color_conversion" function
        """
        unexpected_int = 1
        for unexpected_value in [
            # Unexpected integer is specified as "pipeline" parameter
            unexpected_int,
            # Unexpected integer is specified as nested pipeline
            [{"correct": "dictionary"}, unexpected_int],
        ]:
            with pytest.raises(ValueError):
                patch_color_conversion(pipeline=unexpected_value)  # type: ignore
