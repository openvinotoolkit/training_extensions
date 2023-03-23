# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest

from otx.algorithms.segmentation.adapters.mmseg.datasets.pipelines import (
    LoadAnnotationFromOTXDataset,
    LoadImageFromOTXDataset,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestLoadImageFromOTXDatasetInputParamsValidation:
    @e2e_pytest_unit
    def test_load_image_from_otx_dataset_init_params_validation(self):
        """
        <b>Description:</b>
        Check LoadImageFromOTXDataset object initialization parameters validation

        <b>Input data:</b>
        "to_float32" non-bool parameter

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        LoadImageFromOTXDataset object initialization parameter
        """
        with pytest.raises(ValueError):
            LoadImageFromOTXDataset("unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_load_image_from_otx_dataset_call_params_validation(self):
        """
        <b>Description:</b>
        Check LoadImageFromOTXDataset object "__call__" method input parameters validation

        <b>Input data:</b>
        LoadImageFromOTXDataset object, "results" unexpected type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "__call__" method
        """
        load_image_from_otx_dataset = LoadImageFromOTXDataset()
        unexpected_int = 1
        for unexpected_value in [
            # Unexpected integer is specified as "results" parameter
            unexpected_int,
            # Unexpected integer is specified as "results" dictionary key
            {"result_1": "some results", unexpected_int: "unexpected results"},
        ]:
            with pytest.raises(ValueError):
                load_image_from_otx_dataset.__call__(results=unexpected_value)


class TestLoadAnnotationFromOTXDatasetInputParamsValidation:
    @e2e_pytest_unit
    def test_load_annotation_from_otx_dataset_call_params_validation(self):
        """
        <b>Description:</b>
        Check LoadAnnotationFromOTXDataset object "__call__" method input parameters validation

        <b>Input data:</b>
        LoadAnnotationFromOTXDataset object, "results" unexpected type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "__call__" method
        """
        load_annotation_from_otx_dataset = LoadAnnotationFromOTXDataset()
        unexpected_int = 1
        for unexpected_value in [
            # Unexpected integer is specified as "results" parameter
            unexpected_int,
            # Unexpected integer is specified as "results" dictionary key
            {"result_1": "some results", unexpected_int: "unexpected results"},
        ]:
            with pytest.raises(ValueError):
                load_annotation_from_otx_dataset(results=unexpected_value)
