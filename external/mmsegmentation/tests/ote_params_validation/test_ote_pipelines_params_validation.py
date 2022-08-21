# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest

from ote_sdk.test_suite.e2e_test_system import e2e_pytest_unit
from segmentation_tasks.extension.utils.pipelines import (
    LoadAnnotationFromOTEDataset,
    LoadImageFromOTEDataset,
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
        load_annotation_from_ote_dataset = LoadAnnotationFromOTEDataset()
        unexpected_int = 1
        for unexpected_value in [
            # Unexpected integer is specified as "results" parameter
            unexpected_int,
            # Unexpected integer is specified as "results" dictionary key
            {"result_1": "some results", unexpected_int: "unexpected results"},
        ]:
            with pytest.raises(ValueError):
                load_annotation_from_ote_dataset(results=unexpected_value)
