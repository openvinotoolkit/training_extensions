# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest

from ote_sdk.configuration.configurable_parameters import ConfigurableParameters
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.label_schema import LabelSchemaEntity
from ote_sdk.entities.model import ModelConfiguration, ModelEntity
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.test_suite.e2e_test_system import e2e_pytest_unit
from ote_sdk.tests.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)
from ote_sdk.usecases.tasks.interfaces.optimization_interface import OptimizationType
from segmentation_tasks.apis.segmentation.configuration import OTESegmentationConfig
from segmentation_tasks.apis.segmentation.openvino_task import (
    OpenVINOSegmentationInferencer,
    OTEOpenVinoDataLoader,
    OpenVINOSegmentationTask,
)


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


class MockOpenVinoTask(OpenVINOSegmentationTask):
    def __init__(self):
        pass


class MockOpenVinoInferencer(OpenVINOSegmentationInferencer):
    def __init__(self):
        pass


class TestOpenVINOSegmentationTaskInputParamsValidation:
    @e2e_pytest_unit
    def test_openvino_task_init_params_validation(self):
        """
        <b>Description:</b>
        Check OpenVINOSegmentationTask object initialization parameters validation

        <b>Input data:</b>
        "task_environment" non-TaskEnvironment object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        OpenVINOSegmentationTask object initialization parameter
        """
        with pytest.raises(ValueError):
            OpenVINOSegmentationTask(task_environment="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_openvino_task_infer_params_validation(self):
        """
        <b>Description:</b>
        Check OpenVINOSegmentationTask object "infer" method input parameters validation

        <b>Input data:</b>
        OpenVINOSegmentationTask object. "infer" method unexpected-type input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "infer" method
        """
        task = MockOpenVinoTask()
        correct_values_dict = {"dataset": DatasetEntity()}
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "dataset" parameter
            ("dataset", unexpected_str),
            # Unexpected string is specified as "inference_parameters" parameter
            ("inference_parameters", unexpected_str),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=task.infer,
        )

    @e2e_pytest_unit
    def test_openvino_task_evaluate_params_validation(self):
        """
        <b>Description:</b>
        Check OpenVINOSegmentationTask object "evaluate" method input parameters validation

        <b>Input data:</b>
        OpenVINOSegmentationTask object. "evaluate" method unexpected-type input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "evaluate" method
        """
        result_set = ResultSetEntity(
            model=model(),
            ground_truth_dataset=DatasetEntity(),
            prediction_dataset=DatasetEntity(),
        )
        task = MockOpenVinoTask()
        correct_values_dict = {"output_result_set": result_set}
        unexpected_int = 1
        unexpected_values = [
            # Unexpected integer is specified as "output_result_set" parameter
            ("output_result_set", unexpected_int),
            # Unexpected integer is specified as "evaluation_metric" parameter
            ("evaluation_metric", unexpected_int),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=task.evaluate,
        )

    @e2e_pytest_unit
    def test_openvino_task_deploy_params_validation(self):
        """
        <b>Description:</b>
        Check OpenVINOSegmentationTask object "deploy" method input parameters validation

        <b>Input data:</b>
        OpenVINOSegmentationTask object. "output_model" non-ModelEntity object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "deploy" method
        """
        task = MockOpenVinoTask()
        with pytest.raises(ValueError):
            task.deploy(output_model="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_openvino_task_optimize_params_validation(self):
        """
        <b>Description:</b>
        Check OpenVINOSegmentationTask object "optimize" method input parameters validation

        <b>Input data:</b>
        OpenVINOSegmentationTask object. "optimize" method unexpected-type input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "optimize" method
        """
        task = MockOpenVinoTask()
        correct_values_dict = {
            "optimization_type": OptimizationType.POT,
            "dataset": DatasetEntity(),
            "output_model": model(),
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


class TestOTEOpenVinoDataLoaderInputParamsValidation:
    @e2e_pytest_unit
    def test_openvino_data_loader_init_params_validation(self):
        """
        <b>Description:</b>
        Check OTEOpenVinoDataLoader object initialization parameters validation

        <b>Input data:</b>
        OTEOpenVinoDataLoader object initialization parameters with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        OTEOpenVinoDataLoader object initialization parameter
        """
        classification_inferencer = MockOpenVinoInferencer()
        correct_values_dict = {
            "dataset": DatasetEntity(),
            "inferencer": classification_inferencer,
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "dataset" parameter
            ("dataset", unexpected_str),
            # Unexpected string is specified as "inferencer" parameter
            ("inferencer", unexpected_str),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=OTEOpenVinoDataLoader,
        )

    @e2e_pytest_unit
    def test_openvino_data_loader_getitem_input_params_validation(self):
        """
        <b>Description:</b>
        Check OTEOpenVinoDataLoader object "__getitem__" method input parameters validation

        <b>Input data:</b>
        OTEOpenVinoDataLoader object. "__getitem__" method unexpected-type input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "__getitem__" method
        """
        classification_inferencer = MockOpenVinoInferencer()
        data_loader = OTEOpenVinoDataLoader(
            dataset=DatasetEntity(), inferencer=classification_inferencer
        )
        with pytest.raises(ValueError):
            data_loader.__getitem__(index="unexpected string")  # type: ignore


class TestOpenVINOSegmentationInferencerInputParamsValidation:
    @e2e_pytest_unit
    def test_openvino_segmentation_inferencer_init_params_validation(self):
        """
        <b>Description:</b>
        Check OpenVINOSegmentationInferencer object initialization parameters validation

        <b>Input data:</b>
        OpenVINOSegmentationInferencer object initialization parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        OpenVINOSegmentationInferencer object initialization parameter
        """
        correct_values_dict = {
            "hparams": OTESegmentationConfig("test header"),
            "label_schema": LabelSchemaEntity(),
            "model_file": "some model data",
        }
        unexpected_float = 1.1
        unexpected_values = [
            # Unexpected float is specified as "hparams" parameter
            ("hparams", unexpected_float),
            # Unexpected float is specified as "label_schema" parameter
            ("label_schema", unexpected_float),
            # Unexpected float is specified as "model_file" parameter
            ("model_file", unexpected_float),
            # Unexpected float is specified as "weight_file" parameter
            ("weight_file", unexpected_float),
            # Unexpected float is specified as "device" parameter
            ("device", unexpected_float),
            # Unexpected float is specified as "num_requests" parameter
            ("num_requests", unexpected_float),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=OpenVINOSegmentationInferencer,
        )

    @e2e_pytest_unit
    def test_openvino_segmentation_inferencer_pre_process_params_validation(self):
        """
        <b>Description:</b>
        Check OpenVINOSegmentationInferencer object "pre_process" method input parameters
        validation

        <b>Input data:</b>
        OpenVINOSegmentationInferencer object, "image" non-ndarray object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "pre_process" method
        """
        inferencer = MockOpenVinoInferencer()
        with pytest.raises(ValueError):
            inferencer.pre_process(image="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_openvino_segmentation_inferencer_post_process_params_validation(self):
        """
        <b>Description:</b>
        Check OpenVINOSegmentationInferencer object "post_process" method input parameters
        validation

        <b>Input data:</b>
        OpenVINOSegmentationInferencer object, "post_process" method unexpected-type input
        parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "post_process" method
        """
        inferencer = MockOpenVinoInferencer()
        correct_values_dict = {
            "prediction": {"prediction_1": np.random.rand(2, 2)},
            "metadata": {"metadata_1": "some_data"},
        }
        unexpected_int = 1
        unexpected_values = [
            # Unexpected integer is specified as "prediction" parameter
            ("prediction", unexpected_int),
            # Unexpected integer is specified as "prediction" dictionary key
            ("prediction", {unexpected_int: np.random.rand(2, 2)}),
            # Unexpected integer is specified as "prediction" dictionary value
            ("prediction", {"prediction_1": unexpected_int}),
            # Unexpected integer is specified as "metadata" parameter
            ("metadata", unexpected_int),
            # Unexpected integer is specified as "metadata" dictionary key
            ("metadata", {unexpected_int: "some_data"}),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=inferencer.post_process,
        )

    @e2e_pytest_unit
    def test_openvino_segmentation_inferencer_forward_params_validation(self):
        """
        <b>Description:</b>
        Check OpenVINOSegmentationInferencer object "forward" method input parameters validation

        <b>Input data:</b>
        OpenVINOSegmentationInferencer object, "forward" method unexpected-type input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "forward" method
        """
        inferencer = MockOpenVinoInferencer()
        unexpected_int = 1
        for unexpected_value in [
            # Unexpected integer is specified as "inputs" parameter
            unexpected_int,
            # Unexpected integer is specified as "inputs" dictionary key
            {unexpected_int: np.random.rand(2, 2)},
            # Unexpected integer is specified as "inputs" dictionary value
            {"input_1": unexpected_int},
        ]:
            with pytest.raises(ValueError):
                inferencer.forward(inputs=unexpected_value)  # type: ignore
