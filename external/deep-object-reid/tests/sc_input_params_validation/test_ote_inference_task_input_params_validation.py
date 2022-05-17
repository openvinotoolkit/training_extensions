import pytest
from ote_sdk.configuration.configurable_parameters import ConfigurableParameters
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.label_schema import LabelSchemaEntity
from ote_sdk.entities.model import ModelConfiguration, ModelEntity
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.test_suite.e2e_test_system import e2e_pytest_unit
from ote_sdk.tests.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType

from torchreid_tasks.inference_task import OTEClassificationInferenceTask


class MockClassificationInferenceTask(OTEClassificationInferenceTask):
    def __init__(self):
        pass


class TestOTEClassificationInferenceTaskInputParamsValidation:
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
    def test_ote_classification_inference_task_init_params_validation(self):
        """
        <b>Description:</b>
        Check OTEClassificationInferenceTask object initialization parameters validation

        <b>Input data:</b>
        OTEClassificationInferenceTask object initialization parameters with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        OTEClassificationInferenceTask initialization parameter
        """
        with pytest.raises(ValueError):
            OTEClassificationInferenceTask(task_environment="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_ote_classification_inference_task_infer_params_validation(self):
        """
        <b>Description:</b>
        Check OTEClassificationInferenceTask object "infer" method input parameters validation

        <b>Input data:</b>
        OTEClassificationInferenceTask object. "infer" method unexpected-type input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "infer" method
        """
        task = MockClassificationInferenceTask()
        correct_values_dict = {
            "dataset": DatasetEntity(),
            "inference_parameters": InferenceParameters(),
        }
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
    def test_ote_classification_inference_task_evaluate_params_validation(self):
        """
        <b>Description:</b>
        Check OTEClassificationInferenceTask object "evaluate" method input parameters validation

        <b>Input data:</b>
        OTEClassificationInferenceTask object. "evaluate" method unexpected-type input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "evaluate" method
        """
        task = MockClassificationInferenceTask()
        model = self.model()
        result_set = ResultSetEntity(
            model=model,
            ground_truth_dataset=DatasetEntity(),
            prediction_dataset=DatasetEntity(),
        )
        correct_values_dict = {
            "output_resultset": result_set,
            "evaluation_metric": "metric",
        }
        unexpected_int = 1
        unexpected_values = [
            # Unexpected integer is specified as "output_resultset" parameter
            ("output_resultset", unexpected_int),
            # Unexpected integer is specified as "evaluation_metric" parameter
            ("evaluation_metric", unexpected_int),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=task.evaluate,
        )

    @e2e_pytest_unit
    def test_ote_classification_inference_task_export_params_validation(self):
        """
        <b>Description:</b>
        Check OTEClassificationInferenceTask object "export" method input parameters validation

        <b>Input data:</b>
        OTEClassificationInferenceTask object. "export" method unexpected-type input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "export" method
        """
        task = MockClassificationInferenceTask()
        model = self.model()
        correct_values_dict = {
            "export_type": ExportType.OPENVINO,
            "output_model": model,
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
