from unittest.mock import patch

import pytest

from ote_sdk.entities.label import Domain, LabelEntity
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.shapes.ellipse import Ellipse
from ote_sdk.entities.shapes.polygon import Point, Polygon
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.entities.shapes.shape import Shape, ShapeType
from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements
from ote_sdk.tests.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)
from ote_sdk.utils.time_utils import now


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

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_rectangle_is_full_box_input_parameters_validation(self):
        """
        <b>Description:</b>
        Check Rectangle "is_full_box" method input parameters validation

        <b>Input data:</b>
        Rectangle object, "rectangle" non-Shape object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as "labels" parameter
        for "is_full_box" method
        """
        with pytest.raises(ValueError):
            Rectangle.is_full_box(rectangle="unexpected_str")  # type: ignore

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_rectangle_crop_numpy_array_input_parameters_validation(self):
        """
        <b>Description:</b>
        Check Rectangle "crop_numpy_array" method input parameters validation

        <b>Input data:</b>
        Rectangle object, "data" non-ndarray object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as "data" parameter
        for "is_full_box" method
        """
        rectangle = Rectangle(x1=0.1, y1=0.2, y2=0.3, x2=0.9)
        with pytest.raises(ValueError):
            rectangle.crop_numpy_array(data="unexpected str")  # type: ignore


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestEllipseInputParamsValidation:
    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_ellipse_initialization_parameters_validation(self):
        """
        <b>Description:</b>
        Check Ellipse object initialization parameters validation

        <b>Input data:</b>
        Ellipse object initialization parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as Ellipse initialization
        parameter
        """
        ellipse_label = ScoredLabel(
            label=LabelEntity(name="Ellipse label", domain=Domain.DETECTION)
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
            ("labels", [ellipse_label, unexpected_type_value]),
            # Unexpected string is specified as "modification_date" parameter
            ("modification_date", unexpected_type_value),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=Ellipse,
        )

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_ellipse_get_evenly_distributed_ellipse_coordinates_input_parameters_validation(
        self,
    ):
        """
        <b>Description:</b>
        Check Ellipse "get_evenly_distributed_ellipse_coordinates" method input parameters validation

        <b>Input data:</b>
        Ellipse object, "number_of_coordinates" non-integer object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "number_of_coordinates" parameter for "get_evenly_distributed_ellipse_coordinates" method
        """
        ellipse = Ellipse(x1=0.1, y1=0.2, x2=0.7, y2=0.9)
        with pytest.raises(ValueError):
            ellipse.get_evenly_distributed_ellipse_coordinates(number_of_coordinates=0.8)  # type: ignore


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestPointInputParamsValidation:
    @staticmethod
    def point() -> Point:
        return Point(x=0.1, y=0.1)

    @staticmethod
    def ellipse() -> Ellipse:
        return Ellipse(x1=0.0, y1=0.0, x2=0.8, y2=0.9)

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_point_initialization_parameters_validation(self):
        """
        <b>Description:</b>
        Check Point object initialization parameters validation

        <b>Input data:</b>
        Point object initialization parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as Point initialization
        parameter
        """
        unexpected_type_value = "unexpected str"
        correct_values_dict = {"x": 0.1, "y": 0.1}
        unexpected_values = [
            # Unexpected string is specified as "x" parameter
            ("x", unexpected_type_value),
            # Unexpected string is specified as "y" parameter
            ("y", unexpected_type_value),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=Point,
        )

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_point_normalize_wrt_roi_input_parameters_validation(self):
        """
        <b>Description:</b>
        Check Point "normalize_wrt_roi" method input parameters validation

        <b>Input data:</b>
        Point object, "roi_shape" non-Rectangle object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as "roi_shape" parameter
        for "normalize_wrt_roi" method
        """
        point = self.point()
        with pytest.raises(ValueError):
            point.normalize_wrt_roi(roi_shape=self.ellipse())  # type: ignore

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_point_denormalize_wrt_roi_shape_input_parameters_validation(self):
        """
        <b>Description:</b>
        Check Point "denormalize_wrt_roi_shape" method input parameters validation

        <b>Input data:</b>
        Point object, "roi_shape" non-Rectangle object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as "roi_shape" parameter
        for "denormalize_wrt_roi_shape" method
        """
        point = self.point()
        with pytest.raises(ValueError):
            point.denormalize_wrt_roi_shape(roi_shape=self.ellipse())  # type: ignore


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestPolygonInputParamsValidation:
    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_polygon_initialization_parameters_validation(self):
        """
        <b>Description:</b>
        Check Polygon object initialization parameters validation

        <b>Input data:</b>
        Polygon object initialization parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as Polygon initialization
        parameter
        """
        unexpected_type_value = "unexpected str"
        polygon_label = ScoredLabel(
            label=LabelEntity(name="Polygon label", domain=Domain.DETECTION)
        )
        correct_values_dict = {
            "points": [Point(x=0.2, y=0.2), Point(x=0.2, y=0.8), Point(x=0.6, y=0.2)]
        }
        unexpected_values = [
            # Unexpected string is specified as "points" parameter
            ("points", unexpected_type_value),
            # Unexpected string is specified as nested point
            ("points", [Point(x=0.2, y=0.2), unexpected_type_value]),
            # Unexpected string is specified as "labels" parameter
            ("labels", unexpected_type_value),
            # Unexpected string is specified as nested label
            ("labels", [polygon_label, unexpected_type_value]),
            # Unexpected string is specified as "modification_date" parameter
            ("modification_date", unexpected_type_value),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=Polygon,
        )


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestShapeInputParamsValidation:
    @staticmethod
    def rectangle() -> Shape:
        return Rectangle(x1=0.1, y1=0.2, x2=0.7, y2=0.8)

    @staticmethod
    def shape_label() -> ScoredLabel:
        return ScoredLabel(
            label=LabelEntity(name="Shape Entity label", domain=Domain.DETECTION)
        )

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    @patch("ote_sdk.entities.shapes.shape.Shape.__abstractmethods__", set())
    def test_shape_initialization_parameters_validation(self):
        """
        <b>Description:</b>
        Check Shape object initialization parameters validation

        <b>Input data:</b>
        Shape object initialization parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as Shape initialization
        parameter
        """
        shape_label = self.shape_label()
        correct_values_dict = {
            "type": ShapeType.RECTANGLE,
            "labels": [shape_label],
            "modification_date": now(),
        }
        unexpected_type_value = "unexpected str"
        unexpected_values = [
            # Unexpected string is specified as "type" parameter
            ("type", unexpected_type_value),
            # Unexpected string is specified as "labels" parameter
            ("labels", unexpected_type_value),
            # Unexpected string is specified as nested label
            ("labels", [shape_label, unexpected_type_value]),
            # Unexpected string is specified as "modification_date" parameter
            ("modification_date", unexpected_type_value),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=Shape,
        )

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_shape_intersects_input_parameters_validation(self):
        """
        <b>Description:</b>
        Check Shape "intersects" method input parameters validation

        <b>Input data:</b>
        Shape-type object, "other" non-Shape object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as "other" parameter
        for "intersects" method
        """
        rectangle = self.rectangle()
        with pytest.raises(ValueError):
            rectangle.intersects(other="unexpected string object")  # type: ignore

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_shape_contains_center_input_parameters_validation(self):
        """
        <b>Description:</b>
        Check Shape "contains_center" method input parameters validation

        <b>Input data:</b>
        Shape-type object, "other" non-Shape object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as "other" parameter
        for "contains_center" method
        """
        rectangle = self.rectangle()
        with pytest.raises(ValueError):
            rectangle.contains_center(other="unexpected string object")  # type: ignore

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_shape_get_labels_input_parameters_validation(self):
        """
        <b>Description:</b>
        Check Shape "get_labels" method input parameters validation

        <b>Input data:</b>
        Shape-type object, "include_empty" non-bool object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as "include_empty"
        parameter for "get_labels" method
        """
        rectangle = self.rectangle()
        with pytest.raises(ValueError):
            rectangle.get_labels(include_empty="unexpected string object")  # type: ignore

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_shape_append_label_input_parameters_validation(self):
        """
        <b>Description:</b>
        Check Shape "append_label" method input parameters validation

        <b>Input data:</b>
        Shape-type object, "label" non-LabelEntity object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as "label" parameter for
        "append_label" method
        """
        rectangle = self.rectangle()
        with pytest.raises(ValueError):
            rectangle.append_label(label="unexpected string object")  # type: ignore

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_shape_set_labels_input_parameters_validation(self):
        """
        <b>Description:</b>
        Check Shape "set_labels" method input parameters validation

        <b>Input data:</b>
        Shape-type object, "labels" parameter with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as "labels" parameter for
        "set_labels" method
        """
        shape_label = self.shape_label()
        rectangle = self.rectangle()
        unexpected_type_value = "unexpected str"
        for unexpected_value in (
            unexpected_type_value,
            [shape_label, unexpected_type_value],
        ):
            with pytest.raises(ValueError):
                rectangle.set_labels(labels=unexpected_value)  # type: ignore

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_shape_validate_coordinates_input_parameters_validation(self):
        """
        <b>Description:</b>
        Check Shape "_validate_coordinates" method input parameters validation

        <b>Input data:</b>
        Shape-type object, "x1" and "x2" non-float parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "_validate_coordinates" parameter for "set_labels" method
        """
        rectangle = self.rectangle()
        correct_values_dict = {"x": 0.1, "y": 0.2}
        unexpected_type_value = "unexpected str"
        for key in correct_values_dict:
            incorrect_values_dict = dict(correct_values_dict)
            incorrect_values_dict[key] = unexpected_type_value
            with pytest.raises(ValueError):
                rectangle._validate_coordinates(**incorrect_values_dict)
