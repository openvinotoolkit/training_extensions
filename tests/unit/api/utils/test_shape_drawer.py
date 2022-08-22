# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import datetime
from typing import List

import cv2
import numpy as np
import pytest

from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from otx.api.entities.color import Color
from otx.api.entities.coordinate import Coordinate
from otx.api.entities.id import ID
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.ellipse import Ellipse
from otx.api.entities.shapes.polygon import Point, Polygon
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.utils.shape_drawer import DrawerEntity, Helpers, ShapeDrawer
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements

RANDOM_IMAGE = (np.random.randint(low=0, high=255, size=(1024, 1280, 3))).astype("uint8")


class CommonMethods:
    @staticmethod
    def labels() -> List[LabelEntity]:
        creation_date = datetime.datetime(year=2021, month=12, day=9)
        detection_label = LabelEntity(
            name="Label for Detection",
            domain=Domain.DETECTION,
            color=Color(red=100, green=200, blue=150),
            creation_date=creation_date,
            id=ID("detection_label"),
        )
        segmentation_label = LabelEntity(
            name="Label for Segmentation",
            domain=Domain.DETECTION,
            color=Color(red=50, green=80, blue=200),
            creation_date=creation_date,
            is_empty=True,
            id=ID("segmentation_label"),
        )
        return [detection_label, segmentation_label]

    @staticmethod
    def scored_labels() -> List[ScoredLabel]:
        creation_date = datetime.datetime(year=2021, month=11, day=20)
        classification_label = LabelEntity(
            name="Label for Classification",
            domain=Domain.CLASSIFICATION,
            color=Color(red=200, green=170, blue=90),
            creation_date=creation_date,
            id=ID("classification_label"),
        )
        anomaly_detection_label = LabelEntity(
            name="Label for Anomaly Detection",
            domain=Domain.ANOMALY_DETECTION,
            color=Color(red=100, green=200, blue=190),
            creation_date=creation_date,
            is_empty=True,
            id=ID("anomaly_detection_label"),
        )
        return [ScoredLabel(classification_label), ScoredLabel(anomaly_detection_label)]


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestDrawerEntity:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_drawer_entity(self):
        """
        <b>Description:</b>
        Check DrawerEntity class

        <b>Input data:</b>
        DrawerEntity object

        <b>Expected results:</b>
        Test passes if DrawerEntity object "draw" method raises "NotImplementedError" exception
        """
        entity = Rectangle.generate_full_box()
        labels = CommonMethods.scored_labels()
        with pytest.raises(NotImplementedError):
            DrawerEntity().draw(RANDOM_IMAGE, entity, labels)


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestHelpers:
    @staticmethod
    def generate_expected_image_with_text(
        raw_image,
        text: str,
        initial_text_color: tuple,
        processed_text_color: tuple,
        helpers: Helpers,
        expected_width: int,
        expected_height: int,
        expected_baseline: int,
        text_scale: float,
        thickness: float,
    ) -> np.array:
        processed_image = raw_image.copy()
        helpers.draw_transparent_rectangle(
            img=processed_image,
            x1=int(helpers.cursor_pos.x),
            y1=int(helpers.cursor_pos.y),
            x2=int(helpers.cursor_pos.x + expected_width),
            y2=int(helpers.cursor_pos.y + expected_height),
            color=(
                int(initial_text_color[0]),
                int(initial_text_color[1]),
                int(initial_text_color[2]),
            ),
            alpha=helpers.alpha_labels,
        )
        processed_image = cv2.putText(
            img=processed_image,
            text=text,
            org=(
                helpers.cursor_pos.x + helpers.content_padding,
                helpers.cursor_pos.y + expected_height - helpers.content_padding - expected_baseline,
            ),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=text_scale,
            color=processed_text_color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
        return processed_image

    def generate_image_for_labels(
        self,
        expected_image: np.array,
        labels: list,
        show_labels: bool,
        show_confidence: bool,
        helpers: Helpers,
    ) -> np.array:
        expected_width = 0
        expected_height = 0
        expected_text_scale = helpers.generate_text_scale(expected_image)
        expected_thickness = int(expected_text_scale / 2)
        for label in labels:
            label_text = helpers.generate_text_for_label(label, show_labels, show_confidence)
            label_color = label.color.bgr_tuple
            label_size = cv2.getTextSize(
                label_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=expected_text_scale,
                thickness=expected_thickness,
            )
            label_baseline = label_size[1]
            label_text_width = label_size[0][0]
            label_text_height = label_size[0][1]
            label_width = label_text_width + 2 * helpers.content_padding
            label_height = label_text_height + label_baseline + 2 * helpers.content_padding
            expected_image = self.generate_expected_image_with_text(
                expected_image,
                label_text,
                label_color,
                (255, 255, 255),  # white color
                helpers,
                label_width,
                label_height,
                label_baseline,
                expected_text_scale,
                expected_thickness,
            )
            label_content_width = label_width + helpers.content_margin
            helpers.cursor_pos.x += label_content_width
            helpers.line_height = label_height
            expected_width += label_content_width
            expected_height = label_height
        return expected_image, expected_width, expected_height

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_helpers_initialization(self):
        """
        <b>Description:</b>
        Check Helpers class object initialization

        <b>Input data:</b>
        Helpers class object

        <b>Expected results:</b>
        Test passes if attributes of initialized Helpers class object are equal to expected
        """
        helpers = Helpers()
        assert helpers.alpha_shape == 100 / 256
        assert helpers.alpha_labels == 153 / 256
        assert helpers.assumed_image_width_for_text_scale == 1280
        assert helpers.top_margin == 0.07
        assert helpers.content_padding == 3
        assert helpers.top_left_box_thickness == 1
        assert helpers.content_margin == 2
        assert helpers.label_offset_box_shape == 0
        assert helpers.black == (0, 0, 0)
        assert helpers.white == (255, 255, 255)
        assert helpers.yellow == (255, 255, 0)
        assert helpers.cursor_pos == Coordinate(0, 0)
        assert helpers.line_height == 0

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_helpers_draw_transparent_rectangle(self):
        """
        <b>Description:</b>
        Check Helpers class "draw_transparent_rectangle" method

        <b>Input data:</b>
        Helpers class object

        <b>Expected results:</b>
        Test passes if image array returned by "draw_transparent_rectangle" method is equal to expected

        <b>Steps</b>
        1. Check array returned by "draw_transparent_rectangle" method for rectangle inscribed to image
        2. Check array returned by "draw_transparent_rectangle" method for rectangle equal to image shape
        3. Check array returned by "draw_transparent_rectangle" method for rectangle of out-of-bounds image
        """

        def generate_image_with_rectangle(
            raw_image: np.ndarray,
            start_x: int,
            start_y: int,
            end_x: int,
            end_y: int,
            new_color: tuple,
            new_alpha: float,
        ) -> np.ndarray:
            new_image = raw_image.copy()
            cropped_rectangle = new_image[start_y:end_y, start_x:end_x]
            cropped_rectangle[:] = (new_alpha * np.array(new_color))[np.newaxis, np.newaxis] + (
                1 - new_alpha
            ) * cropped_rectangle
            return new_image

        helpers = Helpers()
        # Checking array returned by "draw_transparent_rectangle" method for rectangle inscribed to image
        image = RANDOM_IMAGE.copy()
        x1, y1, x2, y2 = 200, 100, 1100, 700
        color = (100, 50, 200)
        alpha = 0.8
        expected_image = generate_image_with_rectangle(image, x1, y1, x2 + 1, y2 + 1, color, alpha)
        helpers.draw_transparent_rectangle(image, x1, y1, x2, y2, color, alpha)
        assert np.array_equal(image, expected_image)
        # Checking array returned by "draw_transparent_rectangle" method for rectangle equal to image shape
        image = RANDOM_IMAGE.copy()
        x1, y1, x2, y2 = 0, 0, 1280, 1024
        color = (200, 80, 160)
        alpha = 0.4
        expected_image = generate_image_with_rectangle(image, x1, y1, x2 - 1, y2 - 1, color, alpha)
        helpers.draw_transparent_rectangle(image, x1, y1, x2, y2, color, alpha)
        assert np.array_equal(image, expected_image)
        # Checking array returned by "draw_transparent_rectangle" method for rectangle of out-of-bounds image
        image = RANDOM_IMAGE.copy()
        x2, y2 = 1300, 1100
        color = (70, 90, 20)
        alpha = 0.1
        expected_image = generate_image_with_rectangle(
            image, x1, y1, image.shape[1] - 1, image.shape[0] - 1, color, alpha
        )
        helpers.draw_transparent_rectangle(image, x1, y1, x2, y2, color, alpha)
        assert np.array_equal(image, expected_image)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_helpers_generate_text_scale(self):
        """
        <b>Description:</b>
        Check Helpers class "generate_text_scale" method

        <b>Input data:</b>
        Helpers class object

        <b>Expected results:</b>
        Test passes if value returned by "generate_text_scale" method is equal to expected

        <b>Steps</b>
        1. Check value returned by "generate_text_scale" method for image width less than
        "assumed_image_width_for_text_scale" attribute of Helpers object
        2. Check value returned by "generate_text_scale" method for image width equal to
        "assumed_image_width_for_text_scale" attribute of Helpers object
        3. Check value returned by "generate_text_scale" method for image width more than
        "assumed_image_width_for_text_scale" attribute of Helpers object
        """
        helpers = Helpers()
        # Checking value returned by "generate_text_scale" for image width less than assumed_image_width_for_text_scale
        image = np.random.uniform(low=0.0, high=255.0, size=(16, 10, 3))
        assert helpers.generate_text_scale(image) == 0
        image = np.random.uniform(low=0.0, high=255.0, size=(16, 1279, 3))
        assert helpers.generate_text_scale(image) == 1
        # Checking value returned by "generate_text_scale" for image width less than assumed_image_width_for_text_scale
        image = np.random.uniform(low=0.0, high=255.0, size=(16, 1280, 3))
        assert helpers.generate_text_scale(image) == 1
        # Checking value returned by "generate_text_scale" for image width more than assumed_image_width_for_text_scale
        image = np.random.uniform(low=0.0, high=255.0, size=(16, 1281, 3))
        assert helpers.generate_text_scale(image) == 1
        image = np.random.uniform(low=0.0, high=255.0, size=(16, 2561, 3))
        assert helpers.generate_text_scale(image) == 2

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_helpers_generate_text_for_label(self):
        """
        <b>Description:</b>
        Check Helpers class "generate_text_for_label" method

        <b>Input data:</b>
        Helpers class object

        <b>Expected results:</b>
        Test passes if value returned by "generate_text_for_label" method is equal to expected

        <b>Steps</b>
        1. Check value returned by "generate_text_for_label" for LabelEntity-type object specified as "label" parameter
        2. Check value returned by "generate_text_for_label" for ScoredLabel-type object specified as "label" parameter
        """
        helpers = Helpers()
        # Checking value returned by "generate_text_for_label" for LabelEntity object specified as "label" parameter
        labels = CommonMethods.labels()
        assert helpers.generate_text_for_label(labels[0], True, True) == "Label for Detection"
        assert helpers.generate_text_for_label(labels[0], True, False) == "Label for Detection"
        assert helpers.generate_text_for_label(labels[1], False, True) == ""
        assert helpers.generate_text_for_label(labels[1], False, False) == ""
        # Checking value returned by "generate_text_for_label" for ScoredLabel object specified as "label" parameter
        scored_labels = CommonMethods.scored_labels()
        assert helpers.generate_text_for_label(scored_labels[0], True, True) == "Label for Classification 0%"
        assert helpers.generate_text_for_label(scored_labels[0], True, False) == "Label for Classification"
        assert helpers.generate_text_for_label(scored_labels[1], False, True) == "0%"
        assert helpers.generate_text_for_label(scored_labels[1], False, False) == ""

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_helpers_generate_draw_command_for_text(self):
        """
        <b>Description:</b>
        Check Helpers class "generate_draw_command_for_text" method

        <b>Input data:</b>
        Helpers class object

        <b>Expected results:</b>
        Test passes if tuple returned by "generate_draw_command_for_text" method is equal to expected
        """
        text = "Text to add"
        text_scale = 1.1
        thickness = 2
        expected_label_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=text_scale, thickness=thickness)
        expected_baseline = expected_label_size[1]
        expected_text_width = expected_label_size[0][0]
        expected_text_height = expected_label_size[0][1]
        expected_width = expected_text_width + 2 * 3  # text_width + 2*padding
        expected_content_width = expected_width + 2  # expected_width + margin
        expected_height = expected_text_height + expected_baseline + 2 * 3  # text_height + baseline + 2*padding
        for text_color, expected_text_color in [
            ((150, 40, 100), (255, 255, 255)),  # black text
            ((240, 250, 255), (0, 0, 0)),  # white text
        ]:
            helpers = Helpers()
            draw_command = helpers.generate_draw_command_for_text(text, text_scale, thickness, text_color)
            assert draw_command[1] == expected_content_width
            assert draw_command[2] == expected_height
            image = RANDOM_IMAGE.copy()
            expected_image = self.generate_expected_image_with_text(
                raw_image=image,
                text=text,
                initial_text_color=text_color,
                processed_text_color=expected_text_color,
                helpers=helpers,
                expected_width=expected_width,
                expected_height=expected_height,
                expected_baseline=expected_baseline,
                text_scale=text_scale,
                thickness=thickness,
            )
            actual_image = draw_command[0](image)
            assert np.array_equal(actual_image, expected_image)
            assert helpers.cursor_pos.x == expected_content_width
            assert helpers.line_height == expected_height

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_helpers_generate_draw_command_for_label(self):
        """
        <b>Description:</b>
        Check Helpers class "generate_draw_command_for_labels" method

        <b>Input data:</b>
        Helpers class object

        <b>Expected results:</b>
        Test passes if tuple returned by "generate_draw_command_for_labels" method is equal to expected
        """
        helpers = Helpers()
        expected_helpers = Helpers()
        image = RANDOM_IMAGE.copy()
        expected_image = image.copy()
        labels = CommonMethods.labels() + CommonMethods.scored_labels()
        for show_labels in [True, False]:
            for show_confidence in [True, False]:
                (expected_image, expected_width, expected_height,) = self.generate_image_for_labels(
                    expected_image=expected_image,
                    labels=labels,
                    show_labels=show_labels,
                    show_confidence=show_confidence,
                    helpers=expected_helpers,
                )
                draw_command = helpers.generate_draw_command_for_labels(
                    labels=labels,
                    image=image,
                    show_labels=show_labels,
                    show_confidence=show_confidence,
                )
                assert draw_command[1] == expected_width
                assert draw_command[2] == expected_height
                actual_image = draw_command[0](image)
                assert np.array_equal(actual_image, expected_image)
                assert helpers.cursor_pos.x == expected_helpers.cursor_pos.x
                assert helpers.cursor_pos.y == expected_helpers.cursor_pos.y

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_helpers_draw_flagpole(self):
        """
        <b>Description:</b>
        Check Helpers class "draw_flagpole" method

        <b>Input data:</b>
        Helpers class object

        <b>Expected results:</b>
        Test passes if array returned by "draw_flagpole" method is equal to expected

        <b>Steps</b>
        1. Check array returned by "draw_flagpole" for "coordinates" parameter that fits image borders
        2. Check array returned by "draw_flagpole" for "coordinates" parameter that matches image borders
        3. Check array returned by "draw_flagpole" for "coordinates" parameter that out of image borders
        """
        helpers = Helpers()
        # Checking array returned by "draw_flagpole" for "coordinates" parameter that fits image bounds
        image = RANDOM_IMAGE.copy()
        expected_image = image.copy()
        start_point = Coordinate(1.0, 1.0)
        end_point = Coordinate(1279, 1023)
        actual_image = helpers.draw_flagpole(image, start_point, end_point)
        expected_image = cv2.line(expected_image, (1, 1), (1279, 1023), color=[0, 0, 0], thickness=2)
        assert np.array_equal(actual_image, expected_image)
        # Checking array returned by "draw_flagpole" for "coordinates" parameter that match image borders
        start_point = Coordinate(0.0, 1024.0)
        end_point = Coordinate(1280.0, 0.0)
        actual_image = helpers.draw_flagpole(image, start_point, end_point)
        expected_image = cv2.line(expected_image, (0, 1024), (1280, 0), color=[0, 0, 0], thickness=2)
        assert np.array_equal(actual_image, expected_image)
        # Checking array returned by "draw_flagpole" for "coordinates" parameter that out of image borders
        start_point = Coordinate(0.0, 0.0)
        end_point = Coordinate(1281, 1025)
        actual_image = helpers.draw_flagpole(image, start_point, end_point)
        expected_image = cv2.line(expected_image, (0, 0), (1281, 1025), color=[0, 0, 0], thickness=2)
        assert np.array_equal(actual_image, expected_image)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_helpers_newline(self):
        """
        <b>Description:</b>
        Check Helpers class "newline" method

        <b>Input data:</b>
        Helpers class object

        <b>Expected results:</b>
        Test passes if "cursor_pos.x" and "cursor_pos.y" attributes of Helpers object returned after "newline" method
        are equal to expected
        """
        helpers = Helpers()
        helpers.newline()
        assert helpers.cursor_pos.x == 0  # resets to 0
        assert helpers.cursor_pos.y == 2  # pos.y + content_margin(equal to 2)
        helpers.cursor_pos.x = 10
        helpers.newline()
        assert helpers.cursor_pos.x == 0
        assert helpers.cursor_pos.y == 4

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_helpers_set_cursor_pos(self):
        """
        <b>Description:</b>
        Check Helpers class "set_cursor_pos" method

        <b>Input data:</b>
        Helpers class object

        <b>Expected results:</b>
        Test passes if "cursor_pos.x" and "cursor_pos.y" attributes of Helpers class object returned after
        "set_cursor_pos" method are equal to expected

        <b>Steps</b>
        1. Check cursor_pos.x and cursor_pos.y attributes after "set_cursor_pos" with specified "cursor_pos" parameter
        2. Check cursor_pos.x and cursor_pos.y attributes after "set_cursor_pos" with not specified "cursor_pos"
        parameter
        """
        helpers = Helpers()
        # Checking cursor_pos.x and cursor_pos.y after "set_cursor_pos" method with specified "cursor_pos" parameter
        new_position = Coordinate(60.0, 200.0)
        helpers.set_cursor_pos(new_position)
        assert helpers.cursor_pos.x is new_position.x
        assert helpers.cursor_pos.y is new_position.y
        # Checking cursor_pos.x and cursor_pos.y after "set_cursor_pos" method with non-specified "cursor_pos" parameter
        helpers.set_cursor_pos()
        assert helpers.cursor_pos.x == 0
        assert helpers.cursor_pos.y == 0


class ShapeDrawerParams:
    @staticmethod
    def full_rectangle_labels() -> List[LabelEntity]:
        rectangle_label = LabelEntity(
            name="Full-Rectangle Annotation Label",
            domain=Domain.DETECTION,
            color=Color(100, 200, 60),
            creation_date=datetime.datetime(year=2021, month=12, day=16),
            id=ID("full_rectangle_label_1"),
        )
        other_rectangle_label = LabelEntity(
            name="other Full-Rectangle Annotation Label",
            domain=Domain.SEGMENTATION,
            color=Color(80, 160, 200),
            creation_date=datetime.datetime(year=2021, month=12, day=15),
            id=ID("full_rectangle_label_2"),
        )
        return [rectangle_label, other_rectangle_label]

    def full_rectangle_scored_labels(self) -> List[ScoredLabel]:
        labels = self.full_rectangle_labels()
        return [ScoredLabel(labels[0]), ScoredLabel(labels[1])]

    def full_rectangle_annotation(self) -> Annotation:
        return Annotation(
            shape=Rectangle(x1=0, y1=0, x2=1, y2=1),
            labels=self.full_rectangle_scored_labels(),
            id=ID("full_rectangle_annotation"),
        )

    @staticmethod
    def rectangle_labels() -> List[LabelEntity]:
        rectangle_label = LabelEntity(
            name="Rectangle Annotation Label",
            domain=Domain.DETECTION,
            color=Color(100, 200, 60),
            creation_date=datetime.datetime(year=2021, month=12, day=16),
            id=ID("rectangle_label_1"),
        )
        other_rectangle_label = LabelEntity(
            name="other Rectangle Annotation Label",
            domain=Domain.SEGMENTATION,
            color=Color(80, 160, 200),
            creation_date=datetime.datetime(year=2021, month=12, day=15),
            id=ID("rectangle_label_2"),
        )
        return [rectangle_label, other_rectangle_label]

    def rectangle_scored_labels(self) -> List[ScoredLabel]:
        labels = self.rectangle_labels()
        return [ScoredLabel(labels[0]), ScoredLabel(labels[1])]

    def rectangle_annotation(self) -> Annotation:
        return Annotation(
            shape=Rectangle(x1=0.1, y1=0.4, x2=0.4, y2=0.9),
            labels=self.rectangle_scored_labels(),
            id=ID("rectangle_annotation"),
        )

    @staticmethod
    def polygon_labels() -> List[LabelEntity]:
        polygon_label = LabelEntity(
            name="Polygon Annotation Label",
            domain=Domain.DETECTION,
            color=Color(200, 200, 100),
            creation_date=datetime.datetime(year=2021, month=12, day=16),
            id=ID("polygon_label_1"),
        )
        other_polygon_label = LabelEntity(
            name="other Polygon Annotation Label",
            domain=Domain.SEGMENTATION,
            color=Color(100, 100, 150),
            creation_date=datetime.datetime(year=2021, month=12, day=15),
            id=ID("polygon_label_2"),
        )
        return [polygon_label, other_polygon_label]

    def polygon_scored_labels(self) -> List[ScoredLabel]:
        labels = self.polygon_labels()
        return [ScoredLabel(labels[0]), ScoredLabel(labels[1])]

    def polygon_annotation(self) -> Annotation:
        return Annotation(
            shape=Polygon(
                [
                    Point(0.3, 0.4),
                    Point(0.3, 0.7),
                    Point(0.5, 0.75),
                    Point(0.8, 0.7),
                    Point(0.8, 0.4),
                ]
            ),
            labels=self.polygon_scored_labels(),
            id=ID("polygon_annotation"),
        )

    @staticmethod
    def ellipse_labels() -> List[LabelEntity]:
        ellipse_label = LabelEntity(
            name="Ellipse Annotation Label",
            domain=Domain.DETECTION,
            color=Color(100, 100, 200),
            creation_date=datetime.datetime(year=2021, month=12, day=16),
            id=ID("ellipse_label_1"),
        )
        other_ellipse_label = LabelEntity(
            name="other Ellipse Annotation Label",
            domain=Domain.SEGMENTATION,
            color=Color(200, 80, 150),
            creation_date=datetime.datetime(year=2021, month=12, day=15),
            id=ID("ellipse_label_2"),
        )
        return [ellipse_label, other_ellipse_label]

    def ellipse_scored_labels(self) -> List[ScoredLabel]:
        labels = self.ellipse_labels()
        return [ScoredLabel(labels[0]), ScoredLabel(labels[1])]

    def ellipse_annotation(self) -> Annotation:
        return Annotation(
            shape=Ellipse(x1=0.5, y1=0.0, x2=1.0, y2=0.5),
            labels=self.ellipse_scored_labels(),
            id=ID("ellipse_annotation"),
        )

    def annotation_scene(self) -> AnnotationSceneEntity:
        return AnnotationSceneEntity(
            annotations=[
                self.full_rectangle_annotation(),
                self.rectangle_annotation(),
                self.polygon_annotation(),
                self.ellipse_annotation(),
            ],
            kind=AnnotationSceneKind.ANNOTATION,
            creation_date=datetime.datetime(year=2021, month=12, day=16),
            id=ID("annotation_scene"),
        )


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestShapeDrawer:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_shape_drawer_initialization(self):
        """
        <b>Description:</b>
        Check ShapeDrawer class object initialization

        <b>Input data:</b>
        ShapeDrawer class object with "show_count" and "is_one_label" parameters

        <b>Expected results:</b>
        Test passes if attributes of initialized ShapeDrawer class object are equal to expected

        <b>Steps</b>
        1. Check attributes of ShapeDrawer object initialized with "show_count" parameter is "False" and "is_one_label"
        is "True"
        2. Check attributes of ShapeDrawer object initialized with "show_count" parameter is "True" and "is_one_label"
        is "True"
        3. Check attributes of ShapeDrawer object initialized with "show_count" parameter is "False" and "is_one_label"
        is "False"
        """
        for show_count, is_one_label, show_labels in [
            (False, True, False),
            (True, True, True),
            (False, False, True),
        ]:
            shape_drawer = ShapeDrawer(show_count=show_count, is_one_label=is_one_label)
            assert shape_drawer.show_labels == show_labels
            assert shape_drawer.show_confidence
            assert shape_drawer.show_count == show_count
            assert shape_drawer.is_one_label == is_one_label
            for drawer in shape_drawer.shape_drawers:
                assert drawer.show_labels == show_labels
                assert drawer.show_confidence
            assert shape_drawer.top_left_drawer.show_labels
            assert shape_drawer.top_left_drawer.show_confidence
            assert shape_drawer.top_left_drawer.is_one_label == is_one_label

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_shape_drawer_draw(self):
        """
        <b>Description:</b>
        Check ShapeDrawer class "draw" method

        <b>Input data:</b>
        ShapeDrawer class object with "show_count" and "is_one_label" parameters

        <b>Expected results:</b>
        Test passes if image array returned by "draw" method is equal to expected

        <b>Steps</b>
        1. Check array returned by "draw" method for "show_count" parameter is "True" and "is_one_label" is "True
        2. Check array returned by "draw" method for "show_count" parameter is "False" and "is_one_label" is "False"
        3. Check array returned by "draw" method for "show_count" parameter is "True" and "is_one_label" is "False"
        4. Check array returned by "draw" method for "show_count" parameter is "False" and "is_one_label" is "True"
        """
        annotation_scene = ShapeDrawerParams().annotation_scene()
        annotations = annotation_scene.annotations
        full_rectangle_annotation = annotations[0]
        rectangle_annotation = annotations[1]
        polygon_annotation = annotations[2]
        ellipse_annotation = annotations[3]

        for show_count, is_one_label in [
            (True, True),
            (False, False),
            (True, False),
            (False, True),
        ]:
            image = RANDOM_IMAGE.copy()
            expected_image = image.copy()
            shape_drawer = ShapeDrawer(show_count, is_one_label)
            if not shape_drawer.is_one_label:
                expected_image = shape_drawer.top_left_drawer.draw(expected_image, full_rectangle_annotation, labels=[])
            expected_image = shape_drawer.shape_drawers[0].draw(
                expected_image,
                rectangle_annotation.shape,
                rectangle_annotation.get_labels(),
            )
            expected_image = shape_drawer.shape_drawers[1].draw(
                expected_image,
                polygon_annotation.shape,
                polygon_annotation.get_labels(),
            )
            expected_image = shape_drawer.shape_drawers[2].draw(
                expected_image,
                ellipse_annotation.shape,
                ellipse_annotation.get_labels(),
            )
            if is_one_label:
                expected_image = shape_drawer.top_left_drawer.draw_labels(expected_image, annotation_scene.get_labels())
            if show_count:
                expected_image = shape_drawer.top_left_drawer.draw_annotation_count(expected_image, 3)
            shape_drawer.top_left_drawer.set_cursor_pos()
            actual_image = shape_drawer.draw(image, annotation_scene, [])
            assert np.array_equal(actual_image, expected_image)


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestTopLeftDrawer:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_top_left_drawer_initialization(self):
        """
        <b>Description:</b>
        Check TopLeftDrawer subclass object initialization

        <b>Input data:</b>
        TopLeftDrawer subclass object with "show_count" and "is_one_label" parameters

        <b>Expected results:</b>
        Test passes if attributes of initialized TopLeftDrawer subclass object are equal to expected

        <b>Steps</b>
        1. Check attributes of TopLeftDrawer object initialized for ShapeDrawer with "show_count" parameter is "False"
        and "is_one_label" is "True"
        2. Check attributes of TopLeftDrawer object initialized for ShapeDrawer with "show_count" parameter is "True"
        and "is_one_label" is "True"
        3. Check attributes of TopLeftDrawer object initialized for ShapeDrawer with "show_count" parameter is "False"
        and "is_one_label" is "False"
        """
        for show_count, is_one_label in [(False, True), (True, True), (False, False)]:
            shape_drawer = ShapeDrawer(show_count=show_count, is_one_label=is_one_label)
            assert shape_drawer.top_left_drawer.show_labels
            assert shape_drawer.top_left_drawer.show_confidence
            assert shape_drawer.top_left_drawer.is_one_label == is_one_label

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_top_left_drawer_draw_labels(self):
        """
        <b>Description:</b>
        Check TopLeftDrawer subclass "draw_labels" method

        <b>Input data:</b>
        TopLeftDrawer subclass object with "show_count" and "is_one_label" parameters

        <b>Expected results:</b>
        Test passes if image array returned by "draw_labels" method is equal to expected

        <b>Steps</b>
        1. Check array returned by "draw_labels" method for ShapeDrawer with "show_count" parameter is "False" and
        "is_one_label" is "True"
        2. Check array returned by "draw_labels" method for ShapeDrawer with "show_count" parameter is "True" and
        "is_one_label" is "True"
        3. Check array returned by "draw_labels" method for ShapeDrawer with "show_count" parameter is "False" and
        "is_one_label" is "False"
        """
        labels = ShapeDrawerParams.rectangle_labels() + ShapeDrawerParams().polygon_scored_labels()
        for show_count, is_one_label in [(False, True), (True, True), (False, False)]:
            image = RANDOM_IMAGE.copy()
            expected_image = image.copy()
            shape_drawer = ShapeDrawer(show_count=show_count, is_one_label=is_one_label)
            expected_shape_drawer = ShapeDrawer(show_count=show_count, is_one_label=is_one_label)
            show_confidence = (
                shape_drawer.top_left_drawer.show_confidence if not shape_drawer.top_left_drawer.is_one_label else False
            )
            expected_image = TestHelpers().generate_image_for_labels(
                expected_image,
                labels,
                expected_shape_drawer.top_left_drawer.show_labels,
                show_confidence,
                expected_shape_drawer.top_left_drawer,
            )[0]
            expected_shape_drawer.top_left_drawer.newline()
            actual_image = shape_drawer.top_left_drawer.draw_labels(image, labels)
            assert np.array_equal(actual_image, expected_image)
            assert shape_drawer.top_left_drawer.cursor_pos == expected_shape_drawer.top_left_drawer.cursor_pos

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_top_left_drawer_draw(self):
        """
        <b>Description:</b>
        Check TopLeftDrawer subclass "draw" method

        <b>Input data:</b>
        TopLeftDrawer subclass object with "show_count" and "is_one_label" parameters

        <b>Expected results:</b>
        Test passes if image array returned by "draw" method is equal to expected
        """
        annotation = ShapeDrawerParams().annotation_scene()
        image = RANDOM_IMAGE.copy()
        expected_image = image.copy()
        shape_drawer = ShapeDrawer(True, True)
        expected_shape_drawer = ShapeDrawer(True, True)
        draw_command = expected_shape_drawer.top_left_drawer.generate_draw_command_for_labels(
            annotation.get_labels(), expected_image, True, True
        )[0]
        expected_image = draw_command(expected_image)
        actual_image = shape_drawer.top_left_drawer.draw(image, annotation, [])
        assert np.array_equal(actual_image, expected_image)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_top_left_drawer_draw_annotation_count(self):
        """
        <b>Description:</b>
        Check TopLeftDrawer subclass "draw_annotation_count" method

        <b>Input data:</b>
        TopLeftDrawer subclass object with "show_count" and "is_one_label" parameters

        <b>Expected results:</b>
        Test passes if image array returned by "draw_annotation_count" method is equal to expected
        """
        image = RANDOM_IMAGE.copy()
        expected_image = image.copy()
        shape_drawer = ShapeDrawer(True, True)
        expected_shape_drawer = ShapeDrawer(True, True)
        draw_command = expected_shape_drawer.top_left_drawer.generate_draw_command_for_text(
            "Count: 4", 1.0, 1, (255, 255, 0)
        )[0]
        expected_image = draw_command(expected_image)
        expected_shape_drawer.top_left_drawer.newline()
        actual_image = shape_drawer.top_left_drawer.draw_annotation_count(image, 4)
        assert np.array_equal(actual_image, expected_image)
        assert shape_drawer.top_left_drawer.cursor_pos == expected_shape_drawer.top_left_drawer.cursor_pos


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestRectangleDrawer:
    @staticmethod
    def draw_rectangle_labels(
        image: np.array,
        labels: list,
        shape_drawer: ShapeDrawer,
        rectangle: Rectangle,
        cursor_position: Coordinate,
    ) -> np.ndarray:

        image_copy = image.copy()
        rectangle_drawer = shape_drawer.shape_drawers[0]
        base_color = labels[0].color.bgr_tuple
        x1, y1 = int(rectangle.x1 * image_copy.shape[1]), int(rectangle.y1 * image_copy.shape[0])
        x2, y2 = int(rectangle.x2 * image_copy.shape[1]), int(rectangle.y2 * image_copy.shape[0])
        # Drawing rectangle
        image_copy = rectangle_drawer.draw_transparent_rectangle(
            image_copy, x1, y1, x2, y2, base_color, rectangle_drawer.alpha_shape
        )
        # Drawing rectangle frame
        image_copy = cv2.rectangle(img=image_copy, pt1=(x1, y1), pt2=(x2, y2), color=base_color, thickness=2)
        # Generating draw command to add labels to image
        draw_command, _, _ = rectangle_drawer.generate_draw_command_for_labels(
            labels,
            image_copy,
            rectangle_drawer.show_labels,
            rectangle_drawer.show_confidence,
        )
        rectangle_drawer.set_cursor_pos(cursor_position)
        image_copy = draw_command(image_copy)
        return image_copy

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_rectangle_drawer_initialization(self):
        """
        <b>Description:</b>
        Check RectangleDrawer subclass object initialization

        <b>Input data:</b>
        RectangleDrawer subclass object with "show_count" and "is_one_label" parameters

        <b>Expected results:</b>
        Test passes if attributes of initialized RectangleDrawer subclass object are equal to expected
        """
        for show_count, is_one_label, show_labels in [
            (False, True, False),
            (True, True, True),
            (False, False, True),
        ]:
            shape_drawer = ShapeDrawer(show_count=show_count, is_one_label=is_one_label)
            assert shape_drawer.shape_drawers[0].show_labels == show_labels
            assert shape_drawer.shape_drawers[0].show_confidence

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_rectangle_drawer_draw(self):
        """
        <b>Description:</b>
        Check RectangleDrawer subclass "draw" method

        <b>Input data:</b>
        RectangleDrawer subclass object with "show_count" and "is_one_label" parameters

        <b>Expected results:</b>
        Test passes if image array returned by "draw" method is equal to expected

        <b>Steps</b>
        1. Check array returned by "draw" method without changing labels positions
        2. Check array returned by "draw" method with putting labels to the bottom of drawn rectangle
        3. Check array returned by "draw" method with shifting labels to the left of drawn rectangle
        """
        labels = ShapeDrawerParams().rectangle_scored_labels()
        for rectangle, expected_cursor_position in [
            (  # without changing labels positions
                Rectangle(0.1, 0.3, 0.8, 0.5),
                Coordinate(128, 271),
            ),
            (  # with putting labels to the bottom of drawn rectangle
                Rectangle(0.1, 0.1, 0.9, 0.9),
                Coordinate(128, 102),
            ),
            (  # with shifting labels to the left of drawn rectangle
                Rectangle(0.6, 0.7, 0.9, 0.9),
                Coordinate(61, 680),
            ),
        ]:
            image = RANDOM_IMAGE.copy()
            shape_drawer = ShapeDrawer(True, False)
            expected_shape_drawer = ShapeDrawer(True, False)
            expected_image = self.draw_rectangle_labels(
                image,
                labels,
                expected_shape_drawer,
                rectangle,
                expected_cursor_position,
            )
            actual_image = shape_drawer.shape_drawers[0].draw(image, rectangle, labels)
            assert np.array_equal(actual_image, expected_image)
            assert shape_drawer.top_left_drawer.cursor_pos == expected_shape_drawer.top_left_drawer.cursor_pos


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestEllipseDrawer:
    @staticmethod
    def draw_ellipse_labels(
        image: np.array,
        labels: list,
        shape_drawer: ShapeDrawer,
        ellipse: Ellipse,
        cursor_position: Coordinate,
        flagpole_start: Coordinate,
        flagpole_end: Coordinate,
    ) -> np.ndarray:
        image_copy = image.copy()
        ellipse_shape_drawer = shape_drawer.shape_drawers[2]
        base_color = labels[0].color.bgr_tuple
        if ellipse.width > ellipse.height:
            axes = (
                int(ellipse.major_axis * image_copy.shape[1]),
                int(ellipse.minor_axis * image_copy.shape[0]),
            )
        else:
            axes = (
                int(ellipse.major_axis * image_copy.shape[0]),
                int(ellipse.minor_axis * image_copy.shape[1]),
            )
        center = (
            int(ellipse.x_center * image_copy.shape[1]),
            int(ellipse.y_center * image_copy.shape[0]),
        )
        # Drawing Ellipse on image
        overlay = cv2.ellipse(
            img=image_copy.copy(),
            center=center,
            axes=axes,
            angle=0,
            startAngle=0,
            endAngle=360,
            color=base_color,
            thickness=cv2.FILLED,
        )
        result_without_border = cv2.addWeighted(
            overlay,
            ellipse_shape_drawer.alpha_shape,
            image_copy,
            1 - ellipse_shape_drawer.alpha_shape,
            0,
        )
        # Drawing Ellipse borders on image
        result_with_border = cv2.ellipse(
            img=result_without_border,
            center=center,
            axes=axes,
            angle=0,
            startAngle=0,
            endAngle=360,
            color=base_color,
            lineType=cv2.LINE_AA,
        )
        # Generating draw command to add labels to image
        draw_command = ellipse_shape_drawer.generate_draw_command_for_labels(
            labels,
            image_copy,
            ellipse_shape_drawer.show_labels,
            ellipse_shape_drawer.show_confidence,
        )[0]
        # Getting top-left corner of box around Ellipse
        ellipse_shape_drawer.set_cursor_pos(cursor_position)
        image_copy = draw_command(result_with_border)
        image_copy = ellipse_shape_drawer.draw_flagpole(image_copy, flagpole_start, flagpole_end)
        return image_copy

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_ellipse_drawer_initialization(self):
        """
        <b>Description:</b>
        Check EllipseDrawer subclass object initialization

        <b>Input data:</b>
        EllipseDrawer subclass object with "show_count" and "is_one_label" parameters

        <b>Expected results:</b>
        Test passes if attributes of initialized EllipseDrawer subclass object are equal to expected
        """
        for show_count, is_one_label, show_labels in [
            (False, True, False),
            (True, True, True),
            (False, False, True),
        ]:
            shape_drawer = ShapeDrawer(show_count=show_count, is_one_label=is_one_label)
            assert shape_drawer.shape_drawers[2].show_labels == show_labels
            assert shape_drawer.shape_drawers[2].show_confidence

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_ellipse_drawer_draw(self):
        """
        <b>Description:</b>
        Check EllipseDrawer subclass "draw" method

        <b>Input data:</b>
        EllipseDrawer subclass object with "show_count" and "is_one_label" parameters

        <b>Expected results:</b>
        Test passes if array returned by "draw" method is equal to expected

        <b>Steps</b>
        1. Check array returned by "draw" method without changing labels positions
        2. Check array returned by "draw" method with putting labels to the bottom
        3. Check array returned by "draw" method with shifting labels to the left
        """
        labels = ShapeDrawerParams().ellipse_scored_labels()
        for (ellipse, expected_cursor_position, flagpole_start, flagpole_end,) in [
            (  # without changing labels positions
                Ellipse(0.1, 0.3, 0.8, 0.5),
                Coordinate(128.0, 271.2),
                Coordinate(129.0, 307.2),
                Coordinate(129, 409),
            ),
            (  # with putting labels to the bottom
                Ellipse(0.1, 0.1, 0.8, 0.8),
                Coordinate(128.0, 921.6),
                Coordinate(129.0, 921.6),
                Coordinate(129, 460),
            ),
            (  # with shifting labels to the left
                Ellipse(0.6, 0.7, 0.9, 0.9),
                Coordinate(299, 680.8),
                Coordinate(769.0, 716.8),
                Coordinate(769, 819),
            ),
        ]:
            image = RANDOM_IMAGE.copy()
            shape_drawer = ShapeDrawer(True, False)
            expected_shape_drawer = ShapeDrawer(True, False)
            expected_image = self.draw_ellipse_labels(
                image,
                labels,
                expected_shape_drawer,
                ellipse,
                expected_cursor_position,
                flagpole_start,
                flagpole_end,
            )
            actual_image = shape_drawer.shape_drawers[2].draw(image, ellipse, labels)
            assert np.array_equal(actual_image, expected_image)
            assert shape_drawer.top_left_drawer.cursor_pos == expected_shape_drawer.top_left_drawer.cursor_pos


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestPolygonDrawer:
    @staticmethod
    def draw_polygon_labels(
        image: np.array,
        labels: list,
        shape_drawer: ShapeDrawer,
        polygon: Polygon,
        cursor_position: Coordinate,
        flagpole_start: Coordinate,
        flagpole_end: Coordinate,
    ) -> np.ndarray:
        image_copy = image.copy()
        polygon_drawer = shape_drawer.shape_drawers[1]
        base_color = labels[0].color.bgr_tuple
        # Draw Polygon on the image
        alpha = polygon_drawer.alpha_shape
        contours = np.array(
            [[point.x * image_copy.shape[1], point.y * image_copy.shape[0]] for point in polygon.points],
            dtype=np.int32,
        )
        overlay = cv2.drawContours(
            image=image_copy.copy(),
            contours=[contours],
            contourIdx=-1,
            color=base_color,
            thickness=cv2.FILLED,
        )
        result_without_border = cv2.addWeighted(overlay, alpha, image_copy, 1 - alpha, 0)
        result_with_border = cv2.drawContours(
            image=result_without_border,
            contours=[contours],
            contourIdx=-1,
            color=base_color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        # Generating draw command to add labels to image
        draw_command = polygon_drawer.generate_draw_command_for_labels(
            labels,
            image_copy,
            polygon_drawer.show_labels,
            polygon_drawer.show_confidence,
        )[0]
        # Getting top-left corner of box around Polygon
        polygon_drawer.set_cursor_pos(cursor_position)
        image_copy = draw_command(result_with_border)
        image_copy = polygon_drawer.draw_flagpole(image_copy, flagpole_start, flagpole_end)
        return image_copy

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_polygon_drawer_initialization(self):
        """
        <b>Description:</b>
        Check PolygonDrawer subclass object initialization

        <b>Input data:</b>
        PolygonDrawer subclass object with "show_count" and "is_one_label" parameters

        <b>Expected results:</b>
        Test passes if attributes of initialized PolygonDrawer subclass object are equal to expected
        """
        for show_count, is_one_label, show_labels in [
            (False, True, False),
            (True, True, True),
            (False, False, True),
        ]:
            shape_drawer = ShapeDrawer(show_count=show_count, is_one_label=is_one_label)
            assert shape_drawer.shape_drawers[1].show_labels == show_labels
            assert shape_drawer.shape_drawers[1].show_confidence

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_polygon_drawer_draw(self):
        """
        <b>Description:</b>
        Check PolygonDrawer subclass "draw" method

        <b>Input data:</b>
        PolygonDrawer subclass object with "show_count" and "is_one_label" parameters

        <b>Expected results:</b>
        Test passes if image array returned by "draw" method is equal to expected

        <b>Steps</b>
        1. Check array returned by "draw" method without changing labels positions
        2. Check array returned by "draw" method with putting labels to the bottom
        3. Check array returned by "draw" method with shifting labels to the left
        """
        labels = ShapeDrawerParams().polygon_scored_labels()
        polygon_no_change_labels_position = Polygon(
            [
                Point(0.2, 0.2),
                Point(0.2, 0.6),
                Point(0.4, 0.7),
                Point(0.6, 0.6),
                Point(0.6, 0.2),
            ]
        )
        polygon_put_labels_to_bottom = Polygon(
            [
                Point(0.2, 0.1),
                Point(0.2, 0.6),
                Point(0.4, 0.7),
                Point(0.6, 0.6),
                Point(0.6, 0.1),
            ]
        )
        polygon_shift_labels_to_left = Polygon(
            [
                Point(0.4, 0.2),
                Point(0.4, 0.5),
                Point(0.6, 0.6),
                Point(0.9, 0.5),
                Point(0.9, 0.2),
            ]
        )

        for (polygon, expected_cursor_position, flagpole_start, flagpole_end,) in [
            (  # without changing labels position
                polygon_no_change_labels_position,
                Coordinate(251, 168),
                Coordinate(257, 204),
                Coordinate(257, 204),
            ),
            (  # with putting labels to the bottom
                polygon_put_labels_to_bottom,
                Coordinate(251, 716),
                Coordinate(257, 716),
                Coordinate(257, 102),
            ),
            (  # with shifting labels to the left
                polygon_shift_labels_to_left,
                Coordinate(251, 168),
                Coordinate(513, 204),
                Coordinate(513, 204),
            ),
        ]:
            image = RANDOM_IMAGE.copy()
            shape_drawer = ShapeDrawer(True, False)
            expected_shape_drawer = ShapeDrawer(True, False)
            expected_image = self.draw_polygon_labels(
                image,
                labels,
                expected_shape_drawer,
                polygon,
                expected_cursor_position,
                flagpole_start,
                flagpole_end,
            )
            actual_image = shape_drawer.shape_drawers[1].draw(image, polygon, labels)
            assert np.array_equal(actual_image, expected_image)
            assert shape_drawer.top_left_drawer.cursor_pos == expected_shape_drawer.top_left_drawer.cursor_pos
