import pytest

from ote_sdk.entities.label import Domain, LabelEntity
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements
from ote_sdk.tests.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestRectangleInputParamsValidation:
    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_rectangle_initialization_parameters_validation(self):
        """
        <b>Description:</b>
        Check Rectangle object initialization parameters validation
        <b>Input data:</b>
        Rectangle object initialization parameters
        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as Rectangle
        initialization parameter
        """
        rectangle_label = ScoredLabel(
            label=LabelEntity(name="Rectangle label", domain=Domain.DETECTION)
        )
        unexpected_type_value = "unexpected str"
        correct_values_dict = {"x1": 0.1, "y1": 0.1, "x2": 0.8, "y2": 0.6}
        unexpected_values = [
            # Unexpected string is specified as "x1" parameter
            ("x1", unexpected_type_value),
            # Unexpected string is specified as "y1" parameter
            ("y1", unexpected_type_value),
            # Unexpected string is specified as "x2" parameter
            ("x2", unexpected_type_value),
            # Unexpected string is specified as "y2" parameter
            ("y2", unexpected_type_value),
            # Unexpected string is specified as "labels" parameter
            ("labels", unexpected_type_value),
            # Unexpected string is specified as nested "label"
            ("labels", [rectangle_label, unexpected_type_value]),
            # Unexpected string is specified as "modification_date" parameter
            ("modification_date", unexpected_type_value),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=Rectangle,
        )
