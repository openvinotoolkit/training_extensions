# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest

from otx.algorithms.segmentation.tasks import SegmentationTrainTask
from otx.api.configuration.configurable_parameters import ConfigurableParameters
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.entities.model import ModelConfiguration, ModelEntity
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.api.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)


class MockSegmentationTrainingTask(SegmentationTrainTask):
    def __init__(self):
        pass


class TestSegmentationTrainTaskInputParamsValidation:
    @staticmethod
    def model():
        model_configuration = ModelConfiguration(
            configurable_parameters=ConfigurableParameters(header="header", description="description"),
            label_schema=LabelSchemaEntity(),
        )
        return ModelEntity(train_dataset=DatasetEntity(), configuration=model_configuration)

    @e2e_pytest_unit
    def test_train_task_train_input_params_validation(self):
        """
        <b>Description:</b>
        Check SegmentationTrainTask object "train" method input parameters validation

        <b>Input data:</b>
        SegmentationTrainTask object, "train" method unexpected-type input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "train" method
        """
        task = MockSegmentationTrainingTask()
        correct_values_dict = {
            "dataset": DatasetEntity(),
            "output_model": self.model(),
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "dataset" parameter
            ("dataset", unexpected_str),
            # Unexpected string is specified as "output_model" parameter
            ("output_model", unexpected_str),
            # Unexpected string is specified as "train_parameters" parameter
            ("train_parameters", unexpected_str),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=task.train,
        )

    @e2e_pytest_unit
    def test_train_task_save_model_input_params_validation(self):
        """
        <b>Description:</b>
        Check SegmentationTrainTask object "save_model" method input parameters validation

        <b>Input data:</b>
        SegmentationTrainTask object, "model" non-ModelEntity object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "save_model" method
        """
        task = MockSegmentationTrainingTask()
        with pytest.raises(ValueError):
            task.save_model("unexpected string")  # type: ignore
