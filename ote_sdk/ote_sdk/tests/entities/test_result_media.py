# INTEL CONFIDENTIAL
#
# Copyright (C) 2021 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and
# your use of them is governed by the express license under which they were provided to
# you ("License"). Unless the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit this software or the related documents
# without Intel's prior written permission.
#
# This software and the related documents are provided as is,
# with no express or implied warranties, other than those that are expressly stated
# in the License.

import datetime

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
from ote_sdk.entities.result_media import ResultMediaEntity
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements

RANDOM_IMAGE = np.random.randint(low=0, high=255, size=(32, 64, 3))


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestResultMediaEntity:
    @staticmethod
    def default_result_media_parameters() -> dict:
        rectangle_label = LabelEntity(
            name="Rectangle Annotation Label",
            domain=Domain.DETECTION,
            color=Color(100, 200, 60),
            creation_date=datetime.datetime(year=2021, month=12, day=16),
            id=ID("rectangle_label_1"),
        )
        rectangle_annotation = Annotation(
            shape=Rectangle(x1=0.1, y1=0.4, x2=0.4, y2=0.9),
            labels=[ScoredLabel(rectangle_label)],
            id=ID("rectangle_annotation"),
        )
        annotation_scene = AnnotationSceneEntity(
            annotations=[rectangle_annotation],
            kind=AnnotationSceneKind.ANNOTATION,
            creation_date=datetime.datetime(year=2021, month=12, day=16),
            id=ID("annotation_scene"),
        )
        return {
            "name": "ResultMedia name",
            "type": "Test ResultMedia",
            "annotation_scene": annotation_scene,
            "numpy": RANDOM_IMAGE,
        }

    def optional_result_media_parameters(self) -> dict:
        optional_result_media_parameters = self.default_result_media_parameters()
        roi_label = LabelEntity(
            "ROI label",
            Domain.DETECTION,
            Color(10, 200, 40),
            creation_date=datetime.datetime(year=2021, month=12, day=18),
            id=ID("roi_label_1"),
        )
        roi = Annotation(
            shape=Rectangle(x1=0.3, y1=0.2, x2=0.7, y2=0.6),
            labels=[ScoredLabel(roi_label)],
            id=ID("roi_annotation"),
        )
        result_media_label = LabelEntity(
            "ResultMedia label",
            Domain.CLASSIFICATION,
            Color(200, 60, 100),
            creation_date=datetime.datetime(year=2021, month=12, day=20),
            id=ID("result_media_1"),
        )
        optional_result_media_parameters["roi"] = roi
        optional_result_media_parameters["label"] = result_media_label
        return optional_result_media_parameters

    def result_media(self):
        return ResultMediaEntity(**self.optional_result_media_parameters())

    @staticmethod
    def check_result_media_attributes(
        result_media: ResultMediaEntity, expected_values: dict
    ):
        assert result_media.name == expected_values.get("name")
        assert result_media.type == expected_values.get("type")
        assert result_media.annotation_scene == expected_values.get("annotation_scene")
        assert np.array_equal(result_media.numpy, expected_values.get("numpy"))
        if not expected_values.get("roi"):
            assert isinstance(result_media.roi.shape, Rectangle)
            assert Rectangle.is_full_box(result_media.roi.shape)
        else:
            assert result_media.roi == expected_values.get("roi")
        assert result_media.label == expected_values.get("label")

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_result_media_initialization(self):
        """
        <b>Description:</b>
        Check ResultMediaEntity class object initialization

        <b>Input data:</b>
        ResultMediaEntity class object with specified "name", "type", "annotation_scene", "numpy", "roi"
        and "label" parameters

        <b>Expected results:</b>
        Test passes if attributes of initialized ResultMediaEntity class object are equal to expected

        <b>Steps</b>
        1. Check attributes of ResultMediaEntity class object initialized with default optional parameters
        2. Check attributes of ResultMediaEntity class object initialized with specified optional parameters
        """
        # Checking attributes of ResultMediaEntity class object initialized with default optional parameters
        initialization_params = self.default_result_media_parameters()
        result_media = ResultMediaEntity(**initialization_params)
        self.check_result_media_attributes(result_media, initialization_params)
        # Checking attributes of ResultMediaEntity class object initialized with specified optional parameters
        initialization_params = self.optional_result_media_parameters()
        result_media = ResultMediaEntity(**initialization_params)
        self.check_result_media_attributes(result_media, initialization_params)

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_result_media_repr(self):
        """
        <b>Description:</b>
        Check ResultMediaEntity class object __repr__ method

        <b>Input data:</b>
        ResultMediaEntity class object with specified "name", "type", "annotation_scene", "numpy", "roi"
        and "label" parameters

        <b>Expected results:</b>
        Test passes if value returned by __repr__ method is equal to expected

        <b>Steps</b>
        1. Check value returned by __repr__ method for ResultMediaEntity class object initialized with default optional
        parameters
        2. Check value returned by __repr__ method for ResultMediaEntity class object initialized with specified
        optional parameters
        """
        # Checking __repr__ method for ResultMediaEntity class object initialized with default optional parameters
        initialization_params = self.default_result_media_parameters()
        annotation_scene = initialization_params.get("annotation_scene")
        numpy = initialization_params.get("numpy")
        result_media = ResultMediaEntity(**initialization_params)
        assert repr(result_media) == (
            f"ResultMediaEntity(name=ResultMedia name, type=Test ResultMedia, annotation_scene={annotation_scene}, "
            f"numpy={numpy}, roi={result_media.roi}, label=None)"
        )
        # Checking __repr__ method for ResultMediaEntity class object initialized with specified optional parameters
        initialization_params = self.optional_result_media_parameters()
        annotation_scene = initialization_params.get("annotation_scene")
        numpy = initialization_params.get("numpy")
        roi = initialization_params.get("roi")
        label = initialization_params.get("label")
        result_media = ResultMediaEntity(**initialization_params)
        assert repr(result_media) == (
            f"ResultMediaEntity(name=ResultMedia name, type=Test ResultMedia, annotation_scene={annotation_scene}, "
            f"numpy={numpy}, roi={roi}, label={label})"
        )

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_result_media_shape(self):
        """
        <b>Description:</b>
        Check ResultMediaEntity class object "width" and "height" properties

        <b>Input data:</b>
        ResultMediaEntity class object with specified "name", "type", "annotation_scene", "numpy", "roi"
        and "label" parameters

        <b>Expected results:</b>
        Test passes if values returned by "width" and "height" properties are equal to expected

        <b>Steps</b>
        1. Check values returned by "width" and "height" properties for initialized ResultMediaEntity object
        2. Manually set new value of "numpy" property and check re-check "numpy", "width" and "height" properties
        """
        # Checking values returned by "width" and "height" properties for initialized ResultMediaEntity object
        result_media = self.result_media()
        assert result_media.width == 64
        assert result_media.height == 32
        # Manually setting new value of "numpy" property and re-checking "numpy, "width" and "height" properties
        new_numpy = np.random.uniform(low=0.0, high=255.0, size=(16, 32, 3))
        result_media.numpy = new_numpy
        np.array_equal(result_media.numpy, new_numpy)
        assert result_media.width == 32
        assert result_media.height == 16

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_result_media_eq(self):
        """
        <b>Description:</b>
        Check ResultMediaEntity class object __eq__ method

        <b>Input data:</b>
        ResultMediaEntity class objects with specified "name", "type", "annotation_scene", "numpy", "roi"
        and "label" parameters

        <b>Expected results:</b>
        Test passes if value returned by __eq__ method is equal to expected

        <b>Steps</b>
        1. Check value returned by __eq__ method for comparing equal ResultMediaEntity objects
        2. Check value returned by __eq__ method for comparing ResultMediaEntity objects with unequal
        "name", "type", "label" and "numpy" parameters - expected equality
        3. Check value returned by __eq__ method for comparing ResultMediaEntity objects with unequal
        "annotation_scene" and "roi" parameters - expected inequality
        4. Check value returned by __eq__ method for comparing ResultMediaEntity with different type object
        """
        initialization_params = self.optional_result_media_parameters()
        result_media = ResultMediaEntity(**initialization_params)
        # Comparing equal ResultMediaEntity objects
        equal_result_media = ResultMediaEntity(**initialization_params)
        assert result_media == equal_result_media
        # Comparing ResultMediaEntity objects with unequal "name", "type", "label" and "numpy" parameters,
        # expected equality
        unequal_values = {
            "name": "Unequal name",
            "type": "Unequal type",
            "label": LabelEntity("Unequal label", Domain.CLASSIFICATION),
            "numpy": np.random.uniform(low=0.0, high=255.0, size=(1, 2, 3)),
        }
        for key in unequal_values:
            unequal_params = dict(initialization_params)
            unequal_params[key] = unequal_values.get(key)
            equal_result_media = ResultMediaEntity(**unequal_params)
            assert result_media == equal_result_media
        # Comparing ResultMediaEntity objects with unequal "annotation_scene" and "roi" parameters, expected inequality
        unequal_values = {
            "annotation_scene": AnnotationSceneEntity(
                annotations=[], kind=AnnotationSceneKind.NONE
            ),
            "roi": Rectangle.generate_full_box(),
        }
        for key in unequal_values:
            unequal_params = dict(initialization_params)
            unequal_params[key] = unequal_values.get(key)
            unequal_result_media = ResultMediaEntity(**unequal_params)
            assert result_media != unequal_result_media
        # Comparing ResultMediaEntity with different type object
        assert result_media != str
