# Copyright (C) 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import cv2
import numpy as np
import pytest

from ote_sdk.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from ote_sdk.entities.color import Color
from ote_sdk.entities.id import ID
from ote_sdk.entities.label import Domain, LabelEntity
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.shapes.ellipse import Ellipse
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements
from ote_sdk.usecases.exportable_code.streamer.streamer import MediaType
from ote_sdk.usecases.exportable_code.visualization import Visualizer
from ote_sdk.utils.shape_drawer import ShapeDrawer
from ote_sdk.utils.time_utils import now


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestVisualizer:
    image = np.random.randint(low=0, high=255, size=(480, 640, 3)).astype(np.float32)

    @staticmethod
    def annotation_scene() -> AnnotationSceneEntity:
        creation_date = now()
        annotation_color = Color(red=30, green=180, blue=70)
        other_annotation_color = Color(red=240, green=30, blue=40)
        detection_label = LabelEntity(
            name="detection label",
            domain=Domain.DETECTION,
            color=annotation_color,
            creation_date=creation_date,
            id=ID("detection_1"),
        )
        segmentation_label = LabelEntity(
            name="segmentation label",
            domain=Domain.SEGMENTATION,
            color=annotation_color,
            creation_date=creation_date,
            id=ID("segmentation_1"),
        )
        annotation = Annotation(
            shape=Rectangle(x1=0.1, y1=0.1, x2=0.4, y2=0.5),
            labels=[
                ScoredLabel(detection_label, 0.9),
                ScoredLabel(segmentation_label, 0.8),
            ],
        )

        classification_label = LabelEntity(
            name="classification label",
            domain=Domain.CLASSIFICATION,
            color=other_annotation_color,
            creation_date=creation_date,
            id=ID("classification_1"),
        )
        anomaly_segmentation_label = LabelEntity(
            name="anomaly_segmentation label",
            domain=Domain.ANOMALY_SEGMENTATION,
            color=other_annotation_color,
            creation_date=creation_date,
            id=ID("anomaly_segmentation_1"),
        )
        other_annotation = Annotation(
            shape=Ellipse(x1=0.6, y1=0.4, x2=0.7, y2=0.9),
            labels=[
                ScoredLabel(classification_label, 0.75),
                ScoredLabel(anomaly_segmentation_label, 0.9),
            ],
        )
        annotation_scene = AnnotationSceneEntity(
            annotations=[annotation, other_annotation],
            kind=AnnotationSceneKind.ANNOTATION,
        )
        return annotation_scene

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_visualizer_initialization(self):
        """
        <b>Description:</b>
        Check "Visualizer" class object initialization

        <b>Input data:</b>
        "Visualizer" object with specified attributes

        <b>Expected results:</b>
        Test passes if attributes of initialized "Visualizer" object are equal to expected

        <b>Steps</b>
        1. Check attributes of "Visualizer" object initialized with default optional parameters
        2. Check attributes of "Visualizer" object initialized with default optional parameters except "media_type" is
        set to "IMAGE"
        3. Check attributes of "Visualizer" object initialized with default optional parameters except "media_type" is
        set to "VIDEO"
        4. Check attributes of "Visualizer" object initialized with specified optional parameters
        """

        def check_visualizer_attributes(
            actual_visualizer: Visualizer,
            expected_name: str,
            expected_delay: int,
            expected_show_count: bool,
            expected_is_one_label: bool,
        ):
            assert actual_visualizer.window_name == expected_name
            assert actual_visualizer.delay == expected_delay
            assert isinstance(actual_visualizer.shape_drawer, ShapeDrawer)
            assert actual_visualizer.shape_drawer.show_count == expected_show_count
            assert actual_visualizer.shape_drawer.is_one_label == expected_is_one_label

        # Checking attributes of "Visualizer" initialized with default optional parameters
        visualizer = Visualizer()
        check_visualizer_attributes(
            actual_visualizer=visualizer,
            expected_name="Window",
            expected_delay=0,
            expected_show_count=False,
            expected_is_one_label=False,
        )
        # Checking attributes of "Visualizer" initialized with default optional parameters except "media_type" is set
        # to "IMAGE"
        visualizer = Visualizer(media_type=MediaType.IMAGE)
        check_visualizer_attributes(
            actual_visualizer=visualizer,
            expected_name="Window",
            expected_delay=0,
            expected_show_count=False,
            expected_is_one_label=False,
        )
        # Checking attributes of "Visualizer" initialized with default optional parameters except "media_type" is set
        # to "VIDEO"
        visualizer = Visualizer(media_type=MediaType.VIDEO)
        check_visualizer_attributes(
            actual_visualizer=visualizer,
            expected_name="Window",
            expected_delay=1,
            expected_show_count=False,
            expected_is_one_label=False,
        )
        # Checking attributes of "Visualizer" initialized with specified optional parameters
        visualizer = Visualizer(
            media_type=MediaType.CAMERA,
            window_name="Test Visualizer",
            show_count=True,
            is_one_label=True,
            delay=5,
        )
        check_visualizer_attributes(
            actual_visualizer=visualizer,
            expected_name="Test Visualizer",
            expected_delay=5,
            expected_show_count=True,
            expected_is_one_label=True,
        )

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_visualizer_draw(self):
        """
        <b>Description:</b>
        Check "Visualizer" class "draw" method

        <b>Input data:</b>
        "Visualizer" object with specified attributes, "image" array, "annotation" AnnotationSceneEntity-type object

        <b>Expected results:</b>
        Test passes if array returned by "draw" method is equal to expected
        """
        annotation_scene = self.annotation_scene()
        image = self.image
        expected_image = image.copy()
        expected_image = cv2.cvtColor(expected_image, cv2.COLOR_RGB2BGR)
        shape_drawer = ShapeDrawer(show_count=False, is_one_label=False)
        expected_image = shape_drawer.draw(
            image=expected_image, entity=annotation_scene, labels=[]
        )

        actual_image = Visualizer().draw(image=image, annotation=annotation_scene)
        assert np.array_equal(actual_image, expected_image)

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_visualizer_is_quit(self):
        """
        <b>Description:</b>
        Check "Visualizer" class "is_quit" method

        <b>Input data:</b>
        "Visualizer" object with specified attributes

        <b>Expected results:</b>
        Test passes if bool value returned by "is_quit" method is equal to expected
        """
        visualizer = Visualizer(delay=1001)
        is_quit = visualizer.is_quit()
        assert not is_quit
