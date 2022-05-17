import numpy as np
import pytest
from ote_sdk.test_suite.e2e_test_system import e2e_pytest_unit
from ote_sdk.tests.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)

from torchreid_tasks.model_wrappers.classification import (
    OteClassification,
    sigmoid_numpy,
    softmax_numpy,
    get_hierarchical_predictions,
    get_multiclass_predictions,
    get_multilabel_predictions,
    preprocess_features_for_actmap,
    get_actmap,
)


class MockClassification(OteClassification):
    def __init__(self):
        pass


class TestClassificationFunctionsParamsValidation:
    @e2e_pytest_unit
    def test_preprocess_features_for_actmap_parameters_params_validation(self):
        """
        <b>Description:</b>
        Check "preprocess_features_for_actmap" function input parameters validation

        <b>Input data:</b>
        "features" non-expected type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "preprocess_features_for_actmap" function
        """
        with pytest.raises(ValueError):
            preprocess_features_for_actmap(features=None)  # type: ignore

    @e2e_pytest_unit
    def test_get_actmap_params_validation(self):
        """
        <b>Description:</b>
        Check "get_actmap" function input parameters validation

        <b>Input data:</b>
        "get_actmap" unexpected type parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "get_actmap" function
        """
        correct_values_dict = {
            "features": ["some", "features"],
            "output_res": ("iterable", "object")
        }
        unexpected_values = [
            # Unexpected dictionary is specified as "features" parameter
            ("features", None),
            # Unexpected dictionary is specified as "output_res" parameter
            ("output_res", None),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=get_actmap,
        )

    @e2e_pytest_unit
    def test_sigmoid_numpy_params_validation(self):
        """
        <b>Description:</b>
        Check "sigmoid_numpy" function input parameters validation

        <b>Input data:</b>
        "x" non-nd.array parameter

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "sigmoid_numpy" function
        """
        with pytest.raises(ValueError):
            sigmoid_numpy(x="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_softmax_numpy_params_validation(self):
        """
        <b>Description:</b>
        Check "softmax_numpy" function input parameters validation

        <b>Input data:</b>
        "x" non-nd.array parameter

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "softmax_numpy" function
        """
        with pytest.raises(ValueError):
            softmax_numpy(x="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_get_hierarchical_predictions_params_validation(self):
        """
        <b>Description:</b>
        Check "get_hierarchical_predictions" function input parameters validation

        <b>Input data:</b>
        "get_hierarchical_predictions" unexpected type parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "get_hierarchical_predictions" function
        """
        correct_values_dict = {
            "logits": np.random.randint(low=0, high=255, size=(10, 16, 3)),
            "multihead_class_info": {"multihead": "dictionary"},
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "logits" parameter
            ("logits", unexpected_str),
            # Unexpected string is specified as "multihead_class_info" parameter
            ("multihead_class_info", unexpected_str),
            # Unexpected string is specified as "pos_thr" parameter
            ("pos_thr", unexpected_str),
            # Unexpected string is specified as "activate" parameter
            ("activate", unexpected_str),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=get_hierarchical_predictions,
        )

    @e2e_pytest_unit
    def test_get_multiclass_predictions_params_validation(self):
        """
        <b>Description:</b>
        Check "get_multiclass_predictions" function input parameters validation

        <b>Input data:</b>
        "get_multiclass_predictions" unexpected type parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "get_multiclass_predictions" function
        """
        correct_values_dict = {
            "logits": np.random.randint(low=0, high=255, size=(10, 16, 3)),
            "activate": True,
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "logits" parameter
            ("logits", unexpected_str),
            # Unexpected string is specified as "activate" parameter
            ("activate", unexpected_str),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=get_multiclass_predictions,
        )

    @e2e_pytest_unit
    def test_get_multilabel_predictions_params_validation(self):
        """
        <b>Description:</b>
        Check "get_multilabel_predictions" function input parameters validation

        <b>Input data:</b>
        "get_multilabel_predictions" unexpected type parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "get_multilabel_predictions" function
        """
        correct_values_dict = {
            "logits": np.random.randint(low=0, high=255, size=(10, 16, 3)),
            "activate": True,
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "logits" parameter
            ("logits", unexpected_str),
            # Unexpected string is specified as "pos_thr" parameter
            ("pos_thr", unexpected_str),
            # Unexpected string is specified as "activate" parameter
            ("activate", unexpected_str),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=get_multilabel_predictions,
        )


class TestOteClassificationParamsValidation:
    @e2e_pytest_unit
    def test_ote_classification_preprocess_params_validation(self):
        """
        <b>Description:</b>
        Check OteClassification object "preprocess" method input parameters validation

        <b>Input data:</b>
        OteClassification object. "image" non-ndarray object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "preprocess" method
        """
        classification = MockClassification()
        with pytest.raises(ValueError):
            classification.preprocess(image="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_ote_classification_postprocess_params_validation(self):
        """
        <b>Description:</b>
        Check OteClassification object "postprocess" method input parameters validation

        <b>Input data:</b>
        OteClassification object. "postprocess" method unexpected type parameters

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
        Check OteClassification object "postprocess_aux_outputs" method input parameters validation

        <b>Input data:</b>
        OteClassification object. "postprocess_aux_outputs" method unexpected type parameters

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
