# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np

from otx.algorithms.classification.adapters.openvino.model_wrappers import (
    OTXClassification,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.api.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)


class MockClassification(OTXClassification):
    def __init__(self):
        pass


class TestOTXClassificationParamsValidation:
    @e2e_pytest_unit
    def test_ote_classification_postprocess_params_validation(self):
        """
        <b>Description:</b>
        Check OTXClassification object "postprocess" method input parameters validation

        <b>Input data:</b>
        OTXClassification object. "postprocess" method unexpected type parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "postprocess" method
        """
        classification = MockClassification()
        correct_values_dict = {
            "outputs": {"output_1": np.random.rand(2, 2)},
            "metadata": {"metadata_1": "some_data"},
        }
        unexpected_int = 1
        unexpected_values = [
            # Unexpected integer is specified as "outputs" parameter
            ("outputs", unexpected_int),
            # Unexpected integer is specified as "outputs" dictionary key
            ("outputs", {unexpected_int: np.random.rand(2, 2)}),
            # Unexpected integer is specified as "outputs" dictionary value
            ("outputs", {"output_1": unexpected_int}),
            # Unexpected integer is specified as "metadata" parameter
            ("metadata", unexpected_int),
            # Unexpected integer is specified as "metadata" dictionary key
            ("metadata", {unexpected_int: "some_data"}),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=classification.postprocess,
        )

    @e2e_pytest_unit
    def test_ote_classification_postprocess_aux_outputs_params_validation(self):
        """
        <b>Description:</b>
        Check OTXClassification object "postprocess_aux_outputs" method input parameters validation

        <b>Input data:</b>
        OTXClassification object. "postprocess_aux_outputs" method unexpected type parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "postprocess_aux_outputs" method
        """
        classification = MockClassification()
        correct_values_dict = {
            "outputs": {"output_1": np.random.rand(2, 2)},
            "metadata": {"metadata_1": "some_data"},
        }
        unexpected_int = 1
        unexpected_values = [
            # Unexpected integer is specified as "outputs" parameter
            ("outputs", unexpected_int),
            # Unexpected integer is specified as "outputs" dictionary key
            ("outputs", {unexpected_int: np.random.rand(2, 2)}),
            # Unexpected integer is specified as "outputs" dictionary value
            ("outputs", {"output_1": unexpected_int}),
            # Unexpected integer is specified as "metadata" parameter
            ("metadata", unexpected_int),
            # Unexpected integer is specified as "metadata" dictionary key
            ("metadata", {unexpected_int: "some_data"}),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=classification.postprocess_aux_outputs,
        )
