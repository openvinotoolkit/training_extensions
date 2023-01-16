# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest

from otx.algorithms.classification.configs.base import ClassificationConfig
from otx.algorithms.classification.tasks.openvino import (
    ClassificationOpenVINOInferencer,
    ClassificationOpenVINOTask,
    OTXOpenVinoDataLoader,
)
from otx.api.configuration.configurable_parameters import ConfigurableParameters
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.entities.model import ModelConfiguration, ModelEntity
from otx.api.entities.resultset import ResultSetEntity
from otx.api.usecases.tasks.interfaces.optimization_interface import OptimizationType
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.api.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)


def model():
    model_configuration = ModelConfiguration(
        configurable_parameters=ConfigurableParameters(header="header", description="description"),
        label_schema=LabelSchemaEntity(),
    )
    return ModelEntity(train_dataset=DatasetEntity(), configuration=model_configuration)


class MockOpenVinoTask(ClassificationOpenVINOTask):
    def __init__(self):
        pass


class MockOpenVinoInferencer(ClassificationOpenVINOInferencer):
    def __init__(self):
        pass


class TestClassificationOpenVINOTaskInputParamsValidation:
    @e2e_pytest_unit
    def test_openvino_task_init_params_validation(self):
        """
        <b>Description:</b>
        Check ClassificationOpenVINOTask object initialization parameters validation

        <b>Input data:</b>
        "task_environment" non-TaskEnvironment object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        ClassificationOpenVINOTask object initialization parameter
        """
        with pytest.raises(ValueError):
            ClassificationOpenVINOTask(task_environment="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_openvino_task_infer_params_validation(self):
        """
        <b>Description:</b>
        Check ClassificationOpenVINOTask object "infer" method input parameters validation

        <b>Input data:</b>
        ClassificationOpenVINOTask object. "infer" method unexpected-type input parameters

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
        Check ClassificationOpenVINOTask object "evaluate" method input parameters validation

        <b>Input data:</b>
        ClassificationOpenVINOTask object. "evaluate" method unexpected-type input parameters

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
        Check ClassificationOpenVINOTask object "deploy" method input parameters validation

        <b>Input data:</b>
        ClassificationOpenVINOTask object. "output_model" non-ModelEntity object

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
        Check ClassificationOpenVINOTask object "optimize" method input parameters validation

        <b>Input data:</b>
        ClassificationOpenVINOTask object. "optimize" method unexpected-type input parameters

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


class TestOTXOpenVinoDataLoaderInputParamsValidation:
    @e2e_pytest_unit
    def test_openvino_data_loader_init_params_validation(self):
        """
        <b>Description:</b>
        Check OTXOpenVinoDataLoader object initialization parameters validation

        <b>Input data:</b>
        OTXOpenVinoDataLoader object initialization parameters with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        OTXOpenVinoDataLoader object initialization parameter
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
            class_or_function=OTXOpenVinoDataLoader,
        )

    @e2e_pytest_unit
    def test_openvino_data_loader_getitem_input_params_validation(self):
        """
        <b>Description:</b>
        Check OTXOpenVinoDataLoader object "__getitem__" method input parameters validation

        <b>Input data:</b>
        OTXOpenVinoDataLoader object. "__getitem__" method unexpected-type input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "__getitem__" method
        """
        classification_inferencer = MockOpenVinoInferencer()
        data_loader = OTXOpenVinoDataLoader(dataset=DatasetEntity(), inferencer=classification_inferencer)
        with pytest.raises(ValueError):
            data_loader.__getitem__(index="unexpected string")  # type: ignore


class TestClassificationOpenVINOInferencerInputParamsValidation:
    @e2e_pytest_unit
    def test_openvino_classification_inferencer_init_params_validation(self):
        """
        <b>Description:</b>
        Check ClassificationOpenVINOInferencer object initialization parameters validation

        <b>Input data:</b>
        ClassificationOpenVINOInferencer object initialization parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        ClassificationOpenVINOInferencer object initialization parameter
        """
        correct_values_dict = {
            "hparams": ClassificationConfig("header"),
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
            class_or_function=ClassificationOpenVINOInferencer,
        )

    @e2e_pytest_unit
    def test_openvino_classification_inferencer_pre_process_params_validation(self):
        """
        <b>Description:</b>
        Check ClassificationOpenVINOInferencer object "pre_process" method input parameters
        validation

        <b>Input data:</b>
        ClassificationOpenVINOInferencer object, "image" non-ndarray object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "pre_process" method
        """
        inferencer = MockOpenVinoInferencer()
        with pytest.raises(ValueError):
            inferencer.pre_process(image="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_openvino_classification_inferencer_post_process_params_validation(self):
        """
        <b>Description:</b>
        Check ClassificationOpenVINOInferencer object "post_process" method input parameters
        validation

        <b>Input data:</b>
        ClassificationOpenVINOInferencer object, "post_process" method unexpected-type input
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
    def test_openvino_classification_inferencer_predict_params_validation(self):
        """
        <b>Description:</b>
        Check ClassificationOpenVINOInferencer object "predict" method input parameters
        validation

        <b>Input data:</b>
        ClassificationOpenVINOInferencer object, "image" non-ndarray object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "predict" method
        """
        inferencer = MockOpenVinoInferencer()
        with pytest.raises(ValueError):
            inferencer.predict(image="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_openvino_classification_inferencer_forward_params_validation(self):
        """
        <b>Description:</b>
        Check ClassificationOpenVINOInferencer object "forward" method input parameters validation

        <b>Input data:</b>
        ClassificationOpenVINOInferencer object, "forward" method unexpected-type input parameters

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
