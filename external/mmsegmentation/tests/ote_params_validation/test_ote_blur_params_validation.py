# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np

from mmseg.apis.ote.apis.segmentation.model_wrappers.blur import BlurSegmentation
from ote_sdk.test_suite.e2e_test_system import e2e_pytest_unit
from ote_sdk.tests.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)
from ote_sdk.usecases.adapters.model_adapter import (
    IDataSource,
    ModelAdapter,
)


class MockDataSource(IDataSource):
    def __init__(self):
        pass

    @property
    def data(self):
        return None


class MockBlurSegmentation(BlurSegmentation):
    def __init__(self):
        pass


class TestBlurSegmentationParamsValidation:
    @e2e_pytest_unit
    def test_blur_segmentation_init_params_validation(self):
        """
        <b>Description:</b>
        Check BlurSegmentation object initialization parameters validation

        <b>Input data:</b>
        BlurSegmentation object initialization parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        BlurSegmentation object initialization parameter
        """
        data_source = MockDataSource()
        model_adapter = ModelAdapter(data_source=data_source)
        correct_values_dict = {
            "model_adapter": model_adapter,
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "model_adapter" parameter
            ("model_adapter", unexpected_str),
            # Unexpected string is specified as "configuration" parameter
            ("configuration", unexpected_str),
            # Unexpected string is specified as "preload" parameter
            ("preload", unexpected_str),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=BlurSegmentation,
        )

    @e2e_pytest_unit
    def test_blur_segmentation_postprocess_params_validation(self):
        """
        <b>Description:</b>
        Check BlurSegmentation object "postprocess" method input parameters validation

        <b>Input data:</b>
        BlurSegmentation object, "postprocess" method unexpected-type input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "postprocess" method
        """
        random_array = np.random.randint(low=0, high=255, size=(10, 16, 3))
        blur_segmentation = MockBlurSegmentation()
        correct_values_dict = {
            "outputs": {"output_1": random_array},
            "metadata": {"metadata_1": "some metadata"},
        }
        unexpected_int = 1
        unexpected_values = [
            # Unexpected integer is specified as "outputs" parameter
            ("outputs", unexpected_int),
            # Unexpected integer is specified as "outputs" dictionary key
            ("outputs", {unexpected_int: random_array}),
            # Unexpected integer is specified as "outputs" dictionary value
            ("outputs", {"output_1": unexpected_int}),
            # Unexpected integer is specified as "metadata" parameter
            ("metadata", unexpected_int),
            # Unexpected integer is specified as "metadata" dictionary key
            ("metadata", {unexpected_int: "some metadata"}),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=blur_segmentation.postprocess,
        )
