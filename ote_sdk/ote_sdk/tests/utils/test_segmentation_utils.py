# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import warnings

import cv2
import numpy as np
import pytest

from ote_sdk.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from ote_sdk.entities.dataset_item import DatasetItemEntity
from ote_sdk.entities.id import ID
from ote_sdk.entities.image import Image
from ote_sdk.entities.label import Domain, LabelEntity
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.shapes.ellipse import Ellipse
from ote_sdk.entities.shapes.polygon import Point, Polygon
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements
from ote_sdk.utils.segmentation_utils import (
    create_annotation_from_segmentation_map,
    create_hard_prediction_from_soft_prediction,
    get_subcontours,
    mask_from_annotation,
    mask_from_dataset_item,
)


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestSegmentationUtils:
    @staticmethod
    def rectangle_label():
        return LabelEntity(
            name="Rectangle label",
            domain=Domain.SEGMENTATION,
            id=ID("1_rectangle_label"),
        )

    @staticmethod
    def ellipse_label():
        return LabelEntity(
            name="Ellipse label", domain=Domain.SEGMENTATION, id=ID("3_ellipse_label")
        )

    @staticmethod
    def polygon_label():
        return LabelEntity(
            name="Polygon label", domain=Domain.SEGMENTATION, id=ID("6_polygon_label")
        )

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_mask_from_annotation(self):
        """
        <b>Description:</b>
        Check "mask_from_annotation" function

        <b>Input data:</b>
        List with "Annotation" class objects, list with "LabelEntity" class objects, "width", "height"

        <b>Expected results:</b>
        Test passes if array returned by "mask_from_annotation" function is equal to expected
        """
        rectangle_label = self.rectangle_label()
        ellipse_label = self.ellipse_label()
        polygon_label = self.polygon_label()
        empty_rectangle_label = LabelEntity(
            name="Empty Rectangle label",
            domain=Domain.SEGMENTATION,
            is_empty=True,
            id=ID("2_empty_rectangle_label"),
        )
        empty_ellipse_label = LabelEntity(
            name="Empty Ellipse label",
            domain=Domain.SEGMENTATION,
            is_empty=True,
            id=ID("5_empty_ellipse_label"),
        )
        empty_polygon_label = LabelEntity(
            name="Empty Polygon label",
            domain=Domain.SEGMENTATION,
            is_empty=True,
            id=ID("7_empty_polygon_label"),
        )
        non_annotation_label = LabelEntity(
            name="Non-annotation label",
            domain=Domain.SEGMENTATION,
            id=ID("4_empty_annotation_label"),
        )
        rectangle_annotation = Annotation(
            shape=Rectangle(x1=0.5, y1=0.7, x2=0.9, y2=0.9),
            labels=[ScoredLabel(rectangle_label), ScoredLabel(empty_rectangle_label)],
        )
        ellipse_annotation = Annotation(
            shape=Ellipse(x1=0.5, y1=0.2, x2=0.9, y2=0.4),
            labels=[ScoredLabel(ellipse_label), ScoredLabel(empty_ellipse_label)],
        )
        polygon_shape = Polygon(
            points=[
                Point(x=0.1, y=0.1),
                Point(x=0.1, y=0.3),
                Point(x=0.3, y=0.4),
                Point(x=0.4, y=0.4),
                Point(x=0.4, y=0.1),
            ]
        )
        polygon_annotation = Annotation(
            shape=polygon_shape,
            labels=[ScoredLabel(polygon_label), ScoredLabel(empty_polygon_label)],
        )
        no_labels_annotation = Annotation(
            shape=Rectangle(x1=0.1, y1=0.8, x2=0.2, y2=0.9), labels=[]
        )
        annotations = [
            rectangle_annotation,
            ellipse_annotation,
            polygon_annotation,
            no_labels_annotation,
        ]
        labels = [
            rectangle_label,
            empty_rectangle_label,
            ellipse_label,
            non_annotation_label,
            empty_ellipse_label,
            polygon_label,
            empty_polygon_label,
        ]
        expected_array = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 6, 6, 6, 6, 0, 0, 0, 0, 0],
            [0, 6, 6, 6, 6, 3, 3, 3, 3, 0],
            [0, 6, 6, 6, 6, 3, 3, 3, 3, 0],
            [0, 0, 0, 6, 6, 0, 0, 3, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        ]
        expected_mask = np.expand_dims(expected_array, axis=2)
        mask = mask_from_annotation(
            annotations=annotations, labels=labels, width=10, height=10
        )
        assert np.array_equal(mask, expected_mask)

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_mask_from_dataset_item(self):
        """
        <b>Description:</b>
        Check "mask_from_dataset_item" function

        <b>Input data:</b>
        "DatasetItemEntity" class object, list with "LabelEntity" class objects

        <b>Expected results:</b>
        Test passes if array returned by "mask_from_dataset_item" function is equal to expected
        """
        rectangle_label = self.rectangle_label()
        non_included_label = self.rectangle_label()
        ellipse_label = self.ellipse_label()
        polygon_label = self.polygon_label()
        rectangle_annotation = Annotation(
            shape=Rectangle(x1=0.5, y1=0.7, x2=0.9, y2=0.9),
            labels=[ScoredLabel(rectangle_label), ScoredLabel(non_included_label)],
        )
        ellipse_annotation = Annotation(
            shape=Ellipse(x1=0.5, y1=0.2, x2=0.9, y2=0.4),
            labels=[ScoredLabel(ellipse_label)],
        )
        polygon_shape = Polygon(
            points=[
                Point(x=0.1, y=0.1),
                Point(x=0.1, y=0.3),
                Point(x=0.3, y=0.4),
                Point(x=0.4, y=0.4),
                Point(x=0.4, y=0.1),
            ]
        )
        polygon_annotation = Annotation(
            shape=polygon_shape, labels=[ScoredLabel(polygon_label)]
        )
        image = Image(np.random.randint(low=0, high=255, size=(480, 640, 3)))
        annotation_scene = AnnotationSceneEntity(
            annotations=[rectangle_annotation, ellipse_annotation, polygon_annotation],
            kind=AnnotationSceneKind.ANNOTATION,
        )
        dataset_item = DatasetItemEntity(media=image, annotation_scene=annotation_scene)
        labels = [rectangle_label, ellipse_label, polygon_label]
        expected_mask = mask_from_annotation(
            annotations=dataset_item.get_annotations(),
            labels=labels,
            width=640,
            height=480,
        )
        mask = mask_from_dataset_item(dataset_item=dataset_item, labels=labels)
        assert np.array_equal(mask, expected_mask)

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_create_hard_prediction_from_soft_prediction(self):
        """
        <b>Description:</b>
        Check "create_hard_prediction_from_soft_prediction" function

        <b>Input data:</b>
        "soft_prediction" nd.array, "soft_threshold" value, "blur_strength" value parameters

        <b>Expected results:</b>
        Test passes if array returned by "create_hard_prediction_from_soft_prediction" function is equal to expected

        <b>Steps</b>
        1. Check array returned by "create_hard_prediction_from_soft_prediction" function for 2-dimensional array
        specified as "soft_prediction" parameter
        2. Check array returned by "create_hard_prediction_from_soft_prediction" function for 3-dimensional array
        specified as "soft_prediction" parameter
        3. Check that "ValueError" exception is raised by "create_hard_prediction_from_soft_prediction" function when
        1-dimensional array specified as "soft_prediction" parameter
        """

        def generate_two_dimensional_hard_prediction(prediction, threshold, strength):
            prediction_copy = prediction.copy()
            soft_prediction_blurred = cv2.blur(prediction_copy, (strength, strength))
            hard_prediction = soft_prediction_blurred > threshold
            return hard_prediction

        def generate_three_dimensional_hard_prediction(prediction, threshold, strength):
            prediction_copy = prediction.copy()
            soft_prediction_blurred = cv2.blur(prediction_copy, (strength, strength))
            soft_prediction_blurred[soft_prediction_blurred < threshold] = 0
            hard_prediction = np.argmax(soft_prediction_blurred, axis=2)
            return hard_prediction

        # Checking array returned by "create_hard_prediction_from_soft_prediction" for 2-dimensional array
        # Default value of "blur_strength"
        soft_prediction = np.random.uniform(0, 1.0, size=(10, 15))
        soft_threshold = 0.5
        expected_hard_prediction = generate_two_dimensional_hard_prediction(
            prediction=soft_prediction, threshold=soft_threshold, strength=5
        )
        actual_hard_prediction = create_hard_prediction_from_soft_prediction(
            soft_prediction=soft_prediction, soft_threshold=soft_threshold
        )
        assert np.array_equal(actual_hard_prediction, expected_hard_prediction)
        # Specified value of "blur_strength"
        soft_prediction = np.random.uniform(0, 1.0, size=(10, 15))
        soft_threshold = 0.6
        blur_strength = 4
        expected_hard_prediction = generate_two_dimensional_hard_prediction(
            prediction=soft_prediction, threshold=soft_threshold, strength=blur_strength
        )
        actual_hard_prediction = create_hard_prediction_from_soft_prediction(
            soft_prediction=soft_prediction,
            soft_threshold=soft_threshold,
            blur_strength=blur_strength,
        )
        assert np.array_equal(actual_hard_prediction, expected_hard_prediction)
        # Checking array returned by "create_hard_prediction_from_soft_prediction" for 3-dimensional array
        # Default value of "blur_strength"
        soft_prediction = np.random.uniform(0, 1.0, size=(8, 10, 5))
        soft_threshold = 0.4
        expected_hard_prediction = generate_three_dimensional_hard_prediction(
            prediction=soft_prediction, threshold=soft_threshold, strength=5
        )
        actual_hard_prediction = create_hard_prediction_from_soft_prediction(
            soft_prediction=soft_prediction, soft_threshold=soft_threshold
        )
        assert np.array_equal(actual_hard_prediction, expected_hard_prediction)
        # Checking array returned by "create_hard_prediction_from_soft_prediction" for 3-dimensional array
        # Specified value of "blur_strength"
        soft_prediction = np.random.uniform(0, 1.0, size=(10, 5, 6))
        soft_threshold = 0.5
        blur_strength = 6
        expected_hard_prediction = generate_three_dimensional_hard_prediction(
            prediction=soft_prediction, threshold=soft_threshold, strength=blur_strength
        )
        actual_hard_prediction = create_hard_prediction_from_soft_prediction(
            soft_prediction=soft_prediction,
            soft_threshold=soft_threshold,
            blur_strength=blur_strength,
        )
        assert np.array_equal(actual_hard_prediction, expected_hard_prediction)
        # Checking that "ValueError" exception is raised by "create_hard_prediction_from_soft_prediction" when
        # 1-dimensional array specified as "soft_prediction"
        soft_prediction = np.random.uniform(0, 1.0, size=1)
        with pytest.raises(ValueError):
            create_hard_prediction_from_soft_prediction(soft_prediction, 0.5)

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_get_subcontours(self):
        """
        <b>Description:</b>
        Check "get_subcontours" function

        <b>Input data:</b>
        "Contour" list with coordinates

        <b>Expected results:</b>
        Test passes if list of "Contour" coordinates is equal to expected

        <b>Steps</b>
        1. Check list returned by "get_subcontours" function for closed Contour
        2. Check list returned by "get_subcontours" function for open Contour
        3. Check list returned by "get_subcontours" function for Contour with no intersections
        """
        # Checking list returned by "get_subcontours" for closed Contour
        contour = [
            (0.2, 0.1),  # first rectangle
            (0.2, 0.2),
            (0.2, 0.3),
            (0.3, 0.3),
            (0.3, 0.2),
            (0.3, 0.1),
            (0.2, 0.1),
            (0.3, 0.1),  # second rectangle
            (0.3, 0.2),
            (0.3, 0.3),
            (0.4, 0.3),
            (0.4, 0.2),
            (0.4, 0.1),
            (0.3, 0.1),
            (0.2, 0.1),
        ]
        assert get_subcontours(contour) == [
            [(0.3, 0.1), (0.3, 0.2), (0.3, 0.3), (0.4, 0.3), (0.4, 0.2), (0.4, 0.1)],
            [(0.2, 0.1), (0.2, 0.2), (0.2, 0.3)],
        ]

        # Checking "get_subcontours" for open Contour
        contour = [
            (0.4, 0.4),  # first rectangle
            (0.4, 0.5),
            (0.5, 0.5),
            (0.5, 0.4),
            (0.4, 0.4),
            (0.5, 0.4),  # second rectangle
            (0.5, 0.5),
            (0.6, 0.5),
            (0.6, 0.4),
        ]
        assert get_subcontours(contour) == [
            [(0.4, 0.4), (0.5, 0.4), (0.5, 0.5), (0.6, 0.5), (0.6, 0.4)]
        ]
        # Checking "get_subcontours" for Contour with no intersections
        contour = [(0.1, 0.2), (0.1, 0.2), (0.1, 0.2), (0.1, 0.2)]
        assert get_subcontours(contour) == []

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_create_annotation_from_segmentation_map(self):
        """
        <b>Description:</b>
        Check "create_annotation_from_segmentation_map" function

        <b>Input data:</b>
        "hard_prediction" array, "soft_prediction" array, "label_map" dictionary

        <b>Expected results:</b>
        Test passes if "Annotations" list returned by "create_annotation_from_segmentation_map" function is
        equal to expected

        <b>Steps</b>
        1. Check "Annotations" list returned by "create_annotation_from_segmentation_map" function for 2-dimensional
        prediction arrays
        2. Check "Annotations" list returned by "create_annotation_from_segmentation_map" function for 3-dimensional
        prediction arrays
        3. Check "Annotations" list returned by "create_annotation_from_segmentation_map" function for prediction arrays
        with hole in segmentation mask
        """

        def check_annotation(
            annotation: Annotation,
            expected_points: list,
            expected_label: LabelEntity,
            expected_probability: float,
        ):
            assert isinstance(annotation.shape, Polygon)
            assert annotation.shape.points == expected_points
            annotation_labels = annotation._Annotation__labels  # type: ignore[attr-defined]
            assert len(annotation_labels) == 1
            assert annotation_labels[0].label == expected_label
            assert round(annotation_labels[0].probability, 5) == expected_probability

        # Checking list returned by "create_annotation_from_segmentation_map" for 2-dimensional arrays
        soft_prediction = np.array(
            [
                (0.0, 0.1, 0.8, 0.3, 0.2),
                (0.1, 0.6, 0.7, 0.6, 0.2),
                (0.2, 0.9, 0.8, 0.8, 0.1),
                (0.2, 0.6, 0.9, 0.7, 0.1),
                (0.0, 0.1, 0.2, 0.0, 0.1),
            ]
        )
        hard_prediction = np.array(
            [
                (False, False, True, False, False),
                (False, True, True, True, False),
                (False, True, True, True, False),
                (False, True, True, True, False),
                (False, False, False, False, False),
            ]
        )
        false_label = LabelEntity(name="false_label", domain=Domain.DETECTION)
        true_label = LabelEntity(name="true_label", domain=Domain.DETECTION)
        non_included_label = LabelEntity("label_2", domain=Domain.DETECTION)
        labels = {False: false_label, True: true_label, 2: non_included_label}
        annotations = create_annotation_from_segmentation_map(
            hard_prediction=hard_prediction,
            soft_prediction=soft_prediction,
            label_map=labels,
        )
        assert len(annotations) == 1  # 1 subcontour is created
        check_annotation(
            annotation=annotations[0],
            expected_points=[
                Point(0.4, 0.0),
                Point(0.2, 0.2),
                Point(0.2, 0.4),
                Point(0.2, 0.6),
                Point(0.4, 0.6),
                Point(0.6, 0.6),
                Point(0.6, 0.4),
                Point(0.6, 0.2),
            ],
            expected_label=true_label,
            expected_probability=0.7375,
        )
        # Checking list returned by "create_annotation_from_segmentation_map" for 3-dimensional arrays
        soft_prediction = np.array(
            [
                ([0.8, 0.1, 0.2], [0.9, 0.1, 0.2], [0.3, 0.2, 0.8], [0.1, 0.2, 0.8]),
                ([0.1, 0.8, 0.3], [0.0, 0.8, 0.2], [0.2, 0.1, 0.9], [0.0, 0.2, 0.8]),
                ([0.1, 0.7, 0.3], [0.3, 0.8, 0.2], [0.1, 0.2, 0.8], [0.4, 0.3, 0.7]),
                ([0.0, 1.0, 0.0], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9], [0.2, 0.2, 0.8]),
            ]
        )
        hard_prediction = np.array(
            [(0, 0, 2, 2), (1, 1, 2, 2), (1, 1, 2, 2), (1, 1, 2, 2)]
        )
        class_1_label = LabelEntity(name="class_1_label", domain=Domain.SEGMENTATION)
        class_2_label = LabelEntity(name="class_2_label", domain=Domain.SEGMENTATION)

        labels = {0: false_label, 1: class_1_label, 2: class_2_label}
        annotations = create_annotation_from_segmentation_map(
            hard_prediction=hard_prediction,
            soft_prediction=soft_prediction,
            label_map=labels,
        )
        assert len(annotations) == 2  # 2 subcontours are created
        check_annotation(
            annotation=annotations[0],
            expected_points=[
                Point(0.0, 0.25),
                Point(0.0, 0.5),
                Point(0.0, 0.75),
                Point(0.25, 0.75),
                Point(0.25, 0.5),
                Point(0.25, 0.25),
            ],
            expected_label=class_1_label,
            expected_probability=0.83333,
        )
        check_annotation(
            annotation=annotations[1],
            expected_points=[
                Point(0.5, 0.0),
                Point(0.5, 0.25),
                Point(0.5, 0.5),
                Point(0.5, 0.75),
                Point(0.75, 0.75),
                Point(0.75, 0.5),
                Point(0.75, 0.25),
                Point(0.75, 0.0),
            ],
            expected_label=class_2_label,
            expected_probability=0.8125,
        )
        # Checking list returned by "create_annotation_from_segmentation_map" for prediction arrays with hole in
        # segmentation mask
        soft_prediction = np.array(
            [
                (0.9, 0.85, 0.9, 1.0, 0.85, 0.9, 0.95, 1.0),
                (0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.85),
                (0.9, 0.0, 1.0, 0.9, 0.85, 0.9, 0.0, 0.9),
                (0.85, 0.0, 0.8, 0.0, 0.0, 0.95, 0.0, 0.9),
                (0.9, 0.0, 0.85, 0.0, 0.0, 1.0, 0.0, 0.9),
                (0.85, 0.0, 1.0, 0.9, 0.85, 0.9, 0.0, 0.85),
                (0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9),
                (0.9, 0.95, 1.0, 0.9, 0.95, 0.9, 0.95, 0.95),
            ]
        )
        hard_prediction = np.array(
            [
                (True, True, True, True, True, True, True, True),
                (True, False, False, False, False, False, False, True),
                (True, False, True, True, True, True, False, True),
                (True, False, True, False, False, True, False, True),
                (True, False, True, False, False, True, False, True),
                (True, False, True, True, True, True, False, True),
                (True, False, False, False, False, False, False, True),
                (True, True, True, True, True, True, True, True),
            ]
        )
        labels = {
            False: false_label,
            True: true_label,
            2: non_included_label,
        }
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "The geometry of the segmentation map")
            annotations = create_annotation_from_segmentation_map(
                hard_prediction=hard_prediction,
                soft_prediction=soft_prediction,
                label_map=labels,
            )
        assert len(annotations) == 2  # 2 subcontours are created
        check_annotation(
            annotation=annotations[0],
            expected_points=[
                Point(0.25, 0.25),
                Point(0.25, 0.375),
                Point(0.25, 0.5),
                Point(0.25, 0.625),
                Point(0.375, 0.625),
                Point(0.5, 0.625),
                Point(0.625, 0.625),
                Point(0.625, 0.5),
                Point(0.625, 0.375),
                Point(0.625, 0.25),
                Point(0.5, 0.25),
                Point(0.375, 0.25),
            ],
            expected_label=true_label,
            expected_probability=0.90833,
        )
        check_annotation(
            annotation=annotations[1],
            expected_points=[
                Point(0.0, 0.0),
                Point(0.0, 0.125),
                Point(0.0, 0.25),
                Point(0.0, 0.375),
                Point(0.0, 0.5),
                Point(0.0, 0.625),
                Point(0.0, 0.75),
                Point(0.0, 0.875),
                Point(0.125, 0.875),
                Point(0.25, 0.875),
                Point(0.375, 0.875),
                Point(0.5, 0.875),
                Point(0.625, 0.875),
                Point(0.75, 0.875),
                Point(0.875, 0.875),
                Point(0.875, 0.75),
                Point(0.875, 0.625),
                Point(0.875, 0.5),
                Point(0.875, 0.375),
                Point(0.875, 0.25),
                Point(0.875, 0.125),
                Point(0.875, 0.0),
                Point(0.75, 0.0),
                Point(0.625, 0.0),
                Point(0.5, 0.0),
                Point(0.375, 0.0),
                Point(0.25, 0.0),
                Point(0.125, 0.0),
            ],
            expected_label=true_label,
            expected_probability=0.91071,
        )
