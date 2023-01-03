# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np
import pytest

from otx.algorithms.classification.adapters.deep_object_reid.utils.utils import (
    active_score_from_probs,
    force_fp32,
    get_hierarchical_predictions,
    get_multiclass_predictions,
    get_multilabel_predictions,
    reload_hyper_parameters,
    set_values_as_default,
    sigmoid_numpy,
    softmax_numpy,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.api.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)


class TestDORUtilsParamsValidation:
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

    @e2e_pytest_unit
    def test_active_score_from_probs_parameters_params_validation(self):
        """
        <b>Description:</b>
        Check "active_score_from_probs" function input parameters validation

        <b>Input data:</b>
        "predictions" non-expected type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "active_score_from_probs" function
        """
        with pytest.raises(ValueError):
            active_score_from_probs(predictions=None)  # type: ignore

    @e2e_pytest_unit
    def test_reload_hyper_parameters_params_validation(self):
        """
        <b>Description:</b>
        Check "reload_hyper_parameters" function input parameters validation

        <b>Input data:</b>
        "model_template" non-ModelTemplate parameter

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "reload_hyper_parameters" function
        """
        with pytest.raises(ValueError):
            reload_hyper_parameters(model_template="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_set_values_as_default_parameters_params_validation(self):
        """
        <b>Description:</b>
        Check "set_values_as_default" function input parameters validation

        <b>Input data:</b>
        "parameters" non-ModelTemplate parameter

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "set_values_as_default" function
        """
        with pytest.raises(ValueError):
            set_values_as_default(parameters="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_force_fp32_parameters_params_validation(self):
        """
        <b>Description:</b>
        Check "force_fp32" function input parameters validation

        <b>Input data:</b>
        "model" non-Module parameter

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "force_fp32" function
        """
        with pytest.raises(ValueError):
            with force_fp32(model="unexpected string"):  # type: ignore
                pass
