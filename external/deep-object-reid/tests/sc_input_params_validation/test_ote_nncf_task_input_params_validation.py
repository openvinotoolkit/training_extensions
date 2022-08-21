# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
from ote_sdk.configuration.configurable_parameters import ConfigurableParameters
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.label_schema import LabelSchemaEntity
from ote_sdk.entities.model import ModelConfiguration, ModelEntity
from ote_sdk.test_suite.e2e_test_system import e2e_pytest_unit
from ote_sdk.tests.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType
from ote_sdk.usecases.tasks.interfaces.optimization_interface import OptimizationType

from torchreid_tasks.nncf_task import OTEClassificationNNCFTask


class MockNNCFTask(OTEClassificationNNCFTask):
    def __init__(self):
        pass


class TestNNCFTaskInputParamsValidation:
    @staticmethod
    def model():
        model_configuration = ModelConfiguration(
            configurable_parameters=ConfigurableParameters(
                header="header", description="description"
            ),
            label_schema=LabelSchemaEntity(),
        )
        return ModelEntity(
            train_dataset=DatasetEntity(), configuration=model_configuration
        )

    @e2e_pytest_unit
    def test_ote_nncf_classification_task_init_params_validation(self):
        """
        <b>Description:</b>
        Check OTEClassificationNNCFTask object initialization parameters validation

        <b>Input data:</b>
        OTEClassificationNNCFTask object initialization parameters with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        OTEClassificationNNCFTask object initialization parameter
        """
        with pytest.raises(ValueError):
            OTEClassificationNNCFTask(task_environment="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_ote_nncf_classification_task_optimize_params_validation(self):
        """
        <b>Description:</b>
        Check OTEClassificationNNCFTask object "optimize" method input parameters validation

        <b>Input data:</b>
        OTEClassificationNNCFTask object. "optimize" method unexpected-type input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "optimize" method
        """
        task = MockNNCFTask()
        correct_values_dict = {
            "optimization_type": OptimizationType.NNCF,
            "dataset": DatasetEntity(),
            "output_model": self.model(),
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "optimization_type" parameter
            ("optimization_type", unexpected_str),
            # Unexpected string is specified as "dataset" parameter
            ("dataset", unexpected_str),
            # Unexpected string is specified as "output_model" parameter
            ("output_model", unexpected_str),
            # Unexpected string is specified as "optimization_parameters" parameter
            ("optimization_parameters", unexpected_str),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=task.optimize,
        )

    @e2e_pytest_unit
    def test_ote_nncf_classification_task_save_model_params_validation(self):
        """
        <b>Description:</b>
        Check OTEClassificationNNCFTask object "save_model" method input parameters validation

        <b>Input data:</b>
        OTEClassificationNNCFTask object, "output_model" non-ModelEntity object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "save_model" method
        """
        task = MockNNCFTask()
        with pytest.raises(ValueError):
            task.save_model(output_model="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_ote_nncf_classification_task_export_params_validation(self):
        """
        <b>Description:</b>
        Check OTEClassificationNNCFTask object "export" method input parameters validation

        <b>Input data:</b>
        OTEClassificationNNCFTask object. "export" method unexpected-type input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "export" method
        """
        task = MockNNCFTask()
        correct_values_dict = {
            "export_type": ExportType.OPENVINO,
            "output_model": self.model(),
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "export_type" parameter
            ("export_type", unexpected_str),
            # Unexpected string is specified as "output_model" parameter
            ("output_model", unexpected_str),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=task.export,
        )
