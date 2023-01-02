# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest
from openvino.model_zoo.model_api.models import Model

from otx.algorithms.detection.configs.base import DetectionConfig
from otx.algorithms.detection.tasks.openvino import (
    BaseInferencerWithConverter,
    OpenVINODetectionInferencer,
    OpenVINODetectionTask,
    OpenVINOMaskInferencer,
    OpenVINORotatedRectInferencer,
    OTXOpenVinoDataLoader,
)
from otx.api.configuration.configurable_parameters import ConfigurableParameters
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.entities.model import ModelConfiguration, ModelEntity
from otx.api.entities.resultset import ResultSetEntity
from otx.api.usecases.exportable_code.prediction_to_annotation_converter import (
    DetectionToAnnotationConverter,
)
from otx.api.usecases.tasks.interfaces.optimization_interface import OptimizationType
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.api.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)


class MockOpenVinoTask(OpenVINODetectionTask):
    def __init__(self):
        pass


class MockBaseInferencer(BaseInferencerWithConverter):
    def __init__(self):
        pass


class MockDetectionInferencer(OpenVINODetectionInferencer):
    def __init__(self):
        pass


class MockModel(Model):
    def __init__(self):
        pass

    def preprocess(self):
        pass

    def postprocess(self):
        pass


class TestBaseInferencerWithConverterInputParamsValidation:
    @e2e_pytest_unit
    def test_base_inferencer_with_converter_init_params_validation(self):
        """
        <b>Description:</b>
        Check BaseInferencerWithConverter object initialization parameters validation

        <b>Input data:</b>
        BaseInferencerWithConverter object initialization parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        BaseInferencerWithConverter object initialization parameter
        """
        model = MockModel()
        label = LabelEntity(name="test label", domain=Domain.DETECTION)
        converter = DetectionToAnnotationConverter([label])
        correct_values_dict = {
            "configuration": {"inferencer": "configuration"},
            "model": model,
            "converter": converter,
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "configuration" parameter
            ("configuration", unexpected_str),
            # Unexpected string is specified as "model" parameter
            ("model", unexpected_str),
            # Unexpected string is specified as "converter" parameter
            ("converter", unexpected_str),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=BaseInferencerWithConverter,
        )

    @e2e_pytest_unit
    def test_base_inferencer_with_converter_pre_process_params_validation(self):
        """
        <b>Description:</b>
        Check BaseInferencerWithConverter object "pre_process" method input parameters validation

        <b>Input data:</b>
        BaseInferencerWithConverter object, "image" non-ndarray object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "pre_process" method
        """
        inferencer = MockBaseInferencer()
        with pytest.raises(ValueError):
            inferencer.pre_process(image="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_base_inferencer_with_converter_post_process_params_validation(self):
        """
        <b>Description:</b>
        Check BaseInferencerWithConverter object "post_process" method input parameters validation

        <b>Input data:</b>
        BaseInferencerWithConverter object, "post_process" method unexpected-type input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "post_process" method
        """
        inferencer = MockBaseInferencer()
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
    def test_base_inferencer_with_converter_forward_params_validation(self):
        """
        <b>Description:</b>
        Check BaseInferencerWithConverter object "forward" method input parameters validation

        <b>Input data:</b>
        BaseInferencerWithConverter object, "inputs" unexpected type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "forward" method
        """
        inferencer = MockBaseInferencer()
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


class TestOpenVINODetectionInferencerInputParamsValidation:
    @e2e_pytest_unit
    def test_openvino_detection_inferencer_init_params_validation(self):
        """
        <b>Description:</b>
        Check OpenVINODetectionInferencer object initialization parameters validation

        <b>Input data:</b>
        OpenVINODetectionInferencer object initialization parameters with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        OpenVINODetectionInferencer object initialization parameter
        """
        correct_values_dict = {
            "hparams": DetectionConfig("test header"),
            "label_schema": LabelSchemaEntity(),
            "model_file": "model data",
        }
        unexpected_str = "unexpected string"
        unexpected_int = 1

        unexpected_values = [
            # Unexpected string is specified as "hparams" parameter
            ("hparams", unexpected_str),
            # Unexpected string is specified as "label_schema" parameter
            ("label_schema", unexpected_str),
            # Unexpected integer is specified as "model_file" parameter
            ("model_file", unexpected_int),
            # Unexpected integer is specified as "weight_file" parameter
            ("weight_file", unexpected_int),
            # Unexpected integer is specified as "device" parameter
            ("device", unexpected_int),
            # Unexpected string is specified as "num_requests" parameter
            ("num_requests", unexpected_str),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=OpenVINODetectionInferencer,
        )


class TestOpenVINOMaskInferencerInputParamsValidation:
    @e2e_pytest_unit
    def test_openvino_mask_inferencer_init_params_validation(self):
        """
        <b>Description:</b>
        Check OpenVINOMaskInferencer object initialization parameters validation

        <b>Input data:</b>
        OpenVINOMaskInferencer object initialization parameters with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        OpenVINOMaskInferencer object initialization parameter
        """
        correct_values_dict = {
            "hparams": DetectionConfig("test header"),
            "label_schema": LabelSchemaEntity(),
            "model_file": "model data",
        }
        unexpected_str = "unexpected string"
        unexpected_int = 1

        unexpected_values = [
            # Unexpected string is specified as "hparams" parameter
            ("hparams", unexpected_str),
            # Unexpected string is specified as "label_schema" parameter
            ("label_schema", unexpected_str),
            # Unexpected integer is specified as "model_file" parameter
            ("model_file", unexpected_int),
            # Unexpected integer is specified as "weight_file" parameter
            ("weight_file", unexpected_int),
            # Unexpected integer is specified as "device" parameter
            ("device", unexpected_int),
            # Unexpected string is specified as "num_requests" parameter
            ("num_requests", unexpected_str),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=OpenVINOMaskInferencer,
        )


class TestOpenVINORotatedRectInferencerInputParamsValidation:
    @e2e_pytest_unit
    def test_openvino_rotated_rect_inferencer_init_params_validation(self):
        """
        <b>Description:</b>
        Check OpenVINORotatedRectInferencer object initialization parameters validation

        <b>Input data:</b>
        OpenVINORotatedRectInferencer object initialization parameters with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        OpenVINORotatedRectInferencer object initialization parameter
        """
        correct_values_dict = {
            "hparams": DetectionConfig("test header"),
            "label_schema": LabelSchemaEntity(),
            "model_file": "model data",
        }
        unexpected_str = "unexpected string"
        unexpected_int = 1
        unexpected_values = [
            # Unexpected string is specified as "hparams" parameter
            ("hparams", unexpected_str),
            # Unexpected string is specified as "label_schema" parameter
            ("label_schema", unexpected_str),
            # Unexpected integer is specified as "model_file" parameter
            ("model_file", unexpected_int),
            # Unexpected integer is specified as "weight_file" parameter
            ("weight_file", unexpected_int),
            # Unexpected integer is specified as "device" parameter
            ("device", unexpected_int),
            # Unexpected string is specified as "num_requests" parameter
            ("num_requests", unexpected_str),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=OpenVINORotatedRectInferencer,
        )


class TestOTXOpenVinoDataLoaderInputParamsValidation:
    @staticmethod
    def detection_inferencer(openvino_task):
        return openvino_task.load_inferencer()

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
        correct_values_dict = {
            "dataset": DatasetEntity(),
            "inferencer": MockDetectionInferencer(),
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
        data_loader = OTXOpenVinoDataLoader(dataset=DatasetEntity(), inferencer=MockDetectionInferencer())
        with pytest.raises(ValueError):
            data_loader.__getitem__("unexpected string")  # type: ignore


class TestOpenVINODetectionTaskInputParamsValidation:
    @staticmethod
    def model():
        model_configuration = ModelConfiguration(
            configurable_parameters=ConfigurableParameters(header="header", description="description"),
            label_schema=LabelSchemaEntity(),
        )
        return ModelEntity(train_dataset=DatasetEntity(), configuration=model_configuration)

    @e2e_pytest_unit
    def test_openvino_task_init_params_validation(self):
        """
        <b>Description:</b>
        Check OpenVINODetectionTask object initialization parameters validation

        <b>Input data:</b>
        "task_environment" non-TaskEnvironment object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        OpenVINODetectionTask object initialization parameter
        """
        with pytest.raises(ValueError):
            OpenVINODetectionTask(task_environment="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_openvino_task_infer_params_validation(self):
        """
        <b>Description:</b>
        Check OpenVINODetectionTask object "infer" method input parameters validation

        <b>Input data:</b>
        OpenVINODetectionTask object. "infer" method unexpected-type input parameters

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
        Check OpenVINODetectionTask object "evaluate" method input parameters validation

        <b>Input data:</b>
        OpenVINODetectionTask object. "evaluate" method unexpected-type input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "evaluate" method
        """
        result_set = ResultSetEntity(
            model=self.model(),
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
        Check OpenVINODetectionTask object "deploy" method input parameters validation

        <b>Input data:</b>
        OpenVINODetectionTask object. "output_model" non-ModelEntity object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "deploy" method
        """
        task = MockOpenVinoTask()
        with pytest.raises(ValueError):
            task.deploy("unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_openvino_task_optimize_params_validation(self):
        """
        <b>Description:</b>
        Check OpenVINODetectionTask object "optimize" method input parameters validation

        <b>Input data:</b>
        OpenVINODetectionTask object. "optimize" method unexpected-type input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "optimize" method
        """
        task = MockOpenVinoTask()
        correct_values_dict = {
            "optimization_type": OptimizationType.NNCF,
            "dataset": DatasetEntity(),
            "output_model": self.model(),
            "optimization_parameters": None,
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
