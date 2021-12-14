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
from copy import deepcopy
from typing import List

import numpy as np
import pytest

from ote_sdk.configuration import ConfigurableParameters
from ote_sdk.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from ote_sdk.entities.color import Color
from ote_sdk.entities.dataset_item import DatasetItemEntity
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.id import ID
from ote_sdk.entities.image import Image
from ote_sdk.entities.label import Domain, LabelEntity
from ote_sdk.entities.label_schema import LabelSchemaEntity
from ote_sdk.entities.metadata import MetadataItemEntity
from ote_sdk.entities.model import ModelConfiguration, ModelEntity
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.shapes.ellipse import Ellipse
from ote_sdk.entities.shapes.polygon import Point, Polygon
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.tensor import TensorEntity
from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestDatasetItemEntity:
    @staticmethod
    def generate_random_image() -> Image:
        image = Image(data=np.random.uniform(low=0.0, high=255.0, size=(10, 16, 3)))
        return image

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

    def annotations(self) -> List[Annotation]:
        labels = self.labels()
        rectangle = Rectangle(x1=0.2, y1=0.2, x2=0.6, y2=0.7)
        other_rectangle = Rectangle(x1=0.3, y1=0.2, x2=0.9, y2=0.9)
        detection_annotation = Annotation(
            shape=rectangle,
            labels=[ScoredLabel(label=labels[0])],
            id=ID("detection_annotation_1"),
        )
        segmentation_annotation = Annotation(
            shape=other_rectangle,
            labels=[ScoredLabel(label=labels[1])],
            id=ID("segmentation_annotation_1"),
        )
        return [detection_annotation, segmentation_annotation]

    @staticmethod
    def roi_labels() -> List[LabelEntity]:
        creation_date = datetime.datetime(year=2021, month=12, day=9)
        roi_label = LabelEntity(
            name="ROI label",
            domain=Domain.DETECTION,
            color=Color(red=40, green=180, blue=80),
            creation_date=creation_date,
            id=ID("roi_label_1"),
        )
        other_roi_label = LabelEntity(
            name="Second ROI label",
            domain=Domain.SEGMENTATION,
            color=Color(red=80, green=90, blue=70),
            creation_date=creation_date,
            is_empty=True,
            id=ID("roi_label_2"),
        )
        return [roi_label, other_roi_label]

    def roi_scored_labels(self) -> List[ScoredLabel]:
        roi_labels = self.roi_labels()
        return [ScoredLabel(roi_labels[0]), ScoredLabel(roi_labels[1])]

    def roi(self):
        roi = Annotation(
            shape=Rectangle(
                x1=0.1,
                y1=0.1,
                x2=0.9,
                y2=0.9,
                modification_date=datetime.datetime(year=2021, month=12, day=9),
            ),
            labels=self.roi_scored_labels(),
            id=ID("roi_annotation"),
        )
        return roi

    @staticmethod
    def metadata() -> List[MetadataItemEntity]:
        data = TensorEntity(
            name="test_metadata",
            numpy=np.random.uniform(low=0.0, high=255.0, size=(10, 15, 3)),
        )
        other_data = TensorEntity(
            name="other_metadata",
            numpy=np.random.uniform(low=0.0, high=255.0, size=(10, 15, 3)),
        )
        return [MetadataItemEntity(data=data), MetadataItemEntity(data=other_data)]

    def annotations_entity(self) -> AnnotationSceneEntity:
        return AnnotationSceneEntity(
            annotations=self.annotations(), kind=AnnotationSceneKind.ANNOTATION
        )

    def default_values_dataset_item(self) -> DatasetItemEntity:
        return DatasetItemEntity(
            self.generate_random_image(), self.annotations_entity()
        )

    def dataset_item(self) -> DatasetItemEntity:
        return DatasetItemEntity(
            media=self.generate_random_image(),
            annotation_scene=self.annotations_entity(),
            roi=self.roi(),
            metadata=self.metadata(),
            subset=Subset.TESTING,
        )

    @staticmethod
    def compare_annotations(actual_annotations, expected_annotations) -> None:
        assert len(actual_annotations) == len(expected_annotations)
        for index in range(len(expected_annotations)):
            actual_annotation = actual_annotations[index]
            expected_annotation = expected_annotations[index]
            # Redefining id and modification_date required because of new Annotation objects created after shape
            # denormalize
            actual_annotation.id = expected_annotation.id
            actual_annotation.shape.modification_date = (
                expected_annotation.shape.modification_date
            )
            assert actual_annotation == expected_annotation

    @staticmethod
    def labels_to_add() -> List[LabelEntity]:
        label_to_add = LabelEntity(
            name="Label which will be added",
            domain=Domain.DETECTION,
            color=Color(red=60, green=120, blue=70),
            id=ID("label_to_add_1"),
        )
        other_label_to_add = LabelEntity(
            name="Other label to add",
            domain=Domain.SEGMENTATION,
            color=Color(red=80, green=70, blue=100),
            is_empty=True,
            id=ID("label_to_add_2"),
        )
        return [label_to_add, other_label_to_add]

    def annotations_to_add(self) -> List[Annotation]:
        labels_to_add = self.labels_to_add()
        annotation_to_add = Annotation(
            shape=Rectangle(x1=0.1, y1=0.1, x2=0.7, y2=0.8),
            labels=[ScoredLabel(label=labels_to_add[0])],
            id=ID("added_annotation_1"),
        )
        other_annotation_to_add = Annotation(
            shape=Rectangle(x1=0.2, y1=0.3, x2=0.8, y2=0.9),
            labels=[ScoredLabel(label=labels_to_add[1])],
            id=ID("added_annotation_1"),
        )
        return [annotation_to_add, other_annotation_to_add]

    @staticmethod
    def metadata_item_with_model() -> MetadataItemEntity:
        data = TensorEntity(
            name="appended_metadata_with_model",
            numpy=np.random.uniform(low=0.0, high=255.0, size=(10, 15, 3)),
        )
        configuration = ModelConfiguration(
            configurable_parameters=ConfigurableParameters(header="Test Header"),
            label_schema=LabelSchemaEntity(),
        )
        model = ModelEntity(configuration=configuration, train_dataset=DatasetEntity())
        metadata_item_with_model = MetadataItemEntity(data=data, model=model)
        return metadata_item_with_model

    @staticmethod
    def check_roi_equal_annotation(
        dataset_item: DatasetItemEntity, expected_labels: list, include_empty=False
    ) -> None:
        roi_annotation_in_scene = None
        for annotation in dataset_item.annotation_scene.annotations:
            if annotation == dataset_item.roi:
                assert (
                    annotation.get_labels(include_empty=include_empty)
                    == expected_labels
                )
                roi_annotation_in_scene = annotation
                break
        assert roi_annotation_in_scene

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_item_initialization(self):
        """
        <b>Description:</b>
        Check DatasetItemEntity class object initialization

        <b>Input data:</b>
        DatasetItemEntity class object with specified "media", "annotation_scene", "roi", "metadata" and "subset"
        parameters

        <b>Expected results:</b>
        Test passes if attributes of DatasetItemEntity class object are equal expected

        <b>Steps</b>
        1. Check attributes of DatasetItemEntity object initialized with default optional parameters
        2. Check attributes of DatasetItemEntity object initialized with specified optional parameters
        """

        media = self.generate_random_image()
        annotations_scene = self.annotations_entity()
        # Checking attributes of DatasetItemEntity object initialized with default optional parameters
        default_values_dataset_item = DatasetItemEntity(media, annotations_scene)
        assert default_values_dataset_item.media == media
        assert default_values_dataset_item.annotation_scene == annotations_scene
        assert not default_values_dataset_item.metadata
        assert default_values_dataset_item.subset == Subset.NONE
        # Checking attributes of DatasetItemEntity object initialized with specified optional parameters
        roi = self.roi()
        metadata = self.metadata()
        subset = Subset.TESTING
        specified_values_dataset_item = DatasetItemEntity(
            media, annotations_scene, roi, metadata, subset
        )
        assert specified_values_dataset_item.media == media
        assert specified_values_dataset_item.annotation_scene == annotations_scene
        assert specified_values_dataset_item.roi == roi
        assert specified_values_dataset_item.metadata == metadata
        assert specified_values_dataset_item.subset == subset

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_item_repr(self):
        """
        <b>Description:</b>
        Check DatasetItemEntity class __repr__ method

        <b>Input data:</b>
        DatasetItemEntity class object with specified "media", "annotation_scene", "roi", "metadata" and "subset"
        parameters

        <b>Expected results:</b>
        Test passes if value returned by __repr__ method is equal to expected

        <b>Steps</b>
        1. Check value returned by __repr__ method for  DatasetItemEntity object initialized with default optional
        parameters
        2. Check value returned by __repr__ method for  DatasetItemEntity object initialized with specified optional
        parameters
        """
        media = self.generate_random_image()
        annotation_scene = self.annotations_entity()
        # Checking __repr__ method for DatasetItemEntity object initialized with default optional parameters
        default_values_dataset_item = DatasetItemEntity(media, annotation_scene)
        generated_roi = default_values_dataset_item.roi
        assert repr(default_values_dataset_item) == (
            f"DatasetItemEntity(media=Image(with data, width=16, height=10), "
            f"annotation_scene={annotation_scene}, roi={generated_roi}, "
            f"subset=NONE)"
        )
        # Checking __repr__ method for DatasetItemEntity object initialized with specified optional parameters
        roi = self.roi()
        metadata = self.metadata()
        subset = Subset.TESTING
        specified_values_dataset_item = DatasetItemEntity(
            media, annotation_scene, roi, metadata, subset
        )
        assert repr(specified_values_dataset_item) == (
            f"DatasetItemEntity(media=Image(with data, width=16, height=10), annotation_scene={annotation_scene}, "
            f"roi={roi}, subset=TESTING)"
        )

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_item_roi(self):
        """
        <b>Description:</b>
        Check DatasetItemEntity class "roi" property

        <b>Input data:</b>
        DatasetItemEntity class object with specified "media", "annotation_scene", "roi", "metadata" and "subset"
        parameters

        <b>Expected results:</b>
        Test passes if value returned by "roi" property is equal to expected

        <b>Steps</b>
        1. Check value returned by "roi" property for DatasetItemEntity with specified "roi" parameter
        2. Check value returned by "roi" property for DatasetItemEntity with not specified "roi" parameter
        3. Check value returned by "roi" property for DatasetItemEntity with not specified "roi" parameter but one
        of annotation objects in annotation_scene is equal to full Rectangle
        """
        media = self.generate_random_image()
        annotations = self.annotations()
        annotation_scene = self.annotations_entity()
        roi = self.roi()
        metadata = self.metadata()
        # Checking "roi" property for DatasetItemEntity with specified "roi" parameter
        specified_roi_dataset_item = self.dataset_item()
        assert specified_roi_dataset_item.roi == roi
        # Checking that "roi" property is equal to full_box for DatasetItemEntity with not specified "roi" parameter
        non_specified_roi_dataset_item = DatasetItemEntity(
            media, annotation_scene, metadata=metadata
        )
        default_roi = non_specified_roi_dataset_item.roi.shape
        assert isinstance(default_roi, Rectangle)
        assert Rectangle.is_full_box(default_roi)
        # Checking that "roi" property will be equal to full_box for DatasetItemEntity with not specified "roi" but one
        # of Annotation objects in annotation_scene is equal to full Rectangle
        full_box_label = LabelEntity(
            "Full-box label", Domain.DETECTION, id=ID("full_box_label")
        )
        full_box_annotation = Annotation(
            Rectangle.generate_full_box(), [ScoredLabel(full_box_label)]
        )
        annotations.append(full_box_annotation)
        annotation_scene.annotations.append(full_box_annotation)
        full_box_label_dataset_item = DatasetItemEntity(
            media, annotation_scene, metadata=metadata
        )
        assert full_box_label_dataset_item.roi is full_box_annotation

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_item_roi_numpy(self):
        """
        <b>Description:</b>
        Check DatasetItemEntity class "roi_numpy" method

        <b>Input data:</b>
        DatasetItemEntity class object with specified "media", "annotation_scene", "roi", "metadata" and "subset"
        parameters

        <b>Expected results:</b>
        Test passes if array returned by "roi_numpy" property is equal to expected

        <b>Steps</b>
        1. Check array returned by roi_numpy method with not specified "roi" parameter for DatasetItemEntity with
        "roi" attribute is "None"
        2. Check array returned by roi_numpy method with Rectangle-shape "roi" parameter
        3. Check array returned by roi_numpy method with Ellipse-shape "roi" parameter
        4. Check array returned by roi_numpy method with Polygon-shape "roi" parameter
        5. Check array returned by roi_numpy method with non-specified "roi" parameter for DatasetItemEntity with "roi"
        attribute
        """
        media = self.generate_random_image()
        annotation_scene = self.annotations_entity()
        roi_label = LabelEntity("ROI label", Domain.DETECTION, id=ID("roi_label"))
        dataset_item = DatasetItemEntity(media, annotation_scene)
        # Checking array returned by roi_numpy method with non-specified "roi" parameter for DatasetItemEntity
        # "roi" attribute is "None"
        try:
            assert (dataset_item.roi_numpy() == media.numpy).all()
        except AttributeError:
            raise AssertionError(
                "Unexpected value returned by roi_numpy method for DatasetItem with non-specified roi"
            )
        # Checking array returned by roi_numpy method with specified Rectangle-shape "roi" parameter
        rectangle_roi = Annotation(
            Rectangle(x1=0.2, y1=0.1, x2=0.8, y2=0.9), [ScoredLabel(roi_label)]
        )
        try:
            assert (
                dataset_item.roi_numpy(rectangle_roi) == media.numpy[1:9, 3:13]
            ).all()
        except AttributeError:
            raise AssertionError(
                "Unexpected value returned by roi_numpy method for DatasetItem with Rectangle roi"
            )
        # Checking array returned by roi_numpy method with specified Ellipse-shape "roi" parameter
        ellipse_roi = Annotation(
            Ellipse(x1=0.1, y1=0.0, x2=0.9, y2=0.8), [ScoredLabel(roi_label)]
        )
        try:
            assert (dataset_item.roi_numpy(ellipse_roi) == media.numpy[0:8, 2:14]).all()
        except AttributeError:
            raise AssertionError(
                "Unexpected value returned by roi_numpy method for DatasetItem with Ellipse roi"
            )
        # Checking array returned by roi_numpy method with specified Polygon-shape "roi" parameter
        polygon_roi = Annotation(
            shape=Polygon(
                [
                    Point(0.3, 0.4),
                    Point(0.3, 0.7),
                    Point(0.5, 0.75),
                    Point(0.8, 0.7),
                    Point(0.8, 0.4),
                ]
            ),
            labels=[],
        )
        try:
            assert (dataset_item.roi_numpy(polygon_roi) == media.numpy[4:8, 5:13]).all()
        except AttributeError:
            raise AssertionError(
                "Unexpected value returned by roi_numpy method for DatasetItem with Polygon roi"
            )
        # Checking array returned by roi_numpy method with not specified roi parameter for DatasetItemEntity with "roi"
        # attribute
        roi_specified_dataset_item = DatasetItemEntity(
            media, annotation_scene, self.roi()
        )
        roi_specified_dataset_item.roi_numpy()
        try:
            assert (
                roi_specified_dataset_item.roi_numpy() == media.numpy[1:9, 2:14]
            ).all()
        except AttributeError:
            raise AssertionError(
                "Unexpected value returned by roi_numpy method for DatasetItem with non-specified roi"
            )

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_item_numpy(self):
        """
        <b>Description:</b>
        Check DatasetItemEntity class "numpy" property

        <b>Input data:</b>
        DatasetItemEntity class object with specified "media", "annotation_scene", "roi", "metadata" and "subset"
        parameters

        <b>Expected results:</b>
        Test passes if array returned by "numpy" property is equal to array returned by "roi_numpy" method

        <b>Steps</b>
        1. Check array returned by "numpy" method for DatasetItemEntity with "roi" attribute is "None"
        2. Check array returned by "numpy" method for DatasetItemEntity with specified "roi" attribute
        """
        # Checking array returned by numpy method for DatasetItemEntity with "roi" attribute is "None"
        none_roi_dataset_item = self.default_values_dataset_item()
        try:
            assert (
                none_roi_dataset_item.numpy == none_roi_dataset_item.roi_numpy()
            ).all()
        except AttributeError:
            raise AssertionError(
                "Unexpected value returned by numpy method for DatasetItem with non-specified roi"
            )
        # Checking array returned by numpy method for DatasetItemEntity with specified "roi" attribute
        roi_specified_dataset_item = self.dataset_item()
        try:
            assert (
                roi_specified_dataset_item.numpy
                == roi_specified_dataset_item.roi_numpy()
            ).all()
        except AttributeError:
            raise AssertionError(
                "Unexpected value returned by numpy method for DatasetItem with specified roi"
            )

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_item_width(self):
        """
        <b>Description:</b>
        Check DatasetItemEntity class "width" property

        <b>Input data:</b>
        DatasetItemEntity class object with specified "media", "annotation_scene", "roi", "metadata" and "subset"
        parameters

        <b>Expected results:</b>
        Test passes if value returned by "width" property is equal to expected

        <b>Steps</b>
        1. Check value returned by "width" property for DatasetItemEntity with "roi" attribute is "None"
        2. Check value returned by "width" property for DatasetItemEntity with specified "roi" attribute
        """
        # Checking array returned by "width" property for DatasetItemEntity with "roi" attribute is "None"
        none_roi_dataset_item = self.default_values_dataset_item()
        assert none_roi_dataset_item.width == 16
        # Checking array returned by "width" property for DatasetItemEntity with specified "roi" attribute
        roi_specified_dataset_item = self.dataset_item()
        assert roi_specified_dataset_item.width == 12

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_item_height(self):
        """
        <b>Description:</b>
        Check DatasetItemEntity class "height" property

        <b>Input data:</b>
        DatasetItemEntity class object with specified "media", "annotation_scene", "roi", "metadata" and "subset"
        parameters

        <b>Expected results:</b>
        Test passes if value returned by "height" property is equal to expected

        <b>Steps</b>
        1. Check value returned by "height" property for DatasetItemEntity with "roi" attribute is "None"
        2. Check value returned by "height" property for DatasetItemEntity with specified "roi" attribute
        """
        # Checking array returned by "width" property for DatasetItemEntity with None "roi" attribute
        none_roi_dataset_item = self.default_values_dataset_item()
        assert none_roi_dataset_item.height == 10
        # Checking array returned by "width" property for DatasetItemEntity with specified "roi" attribute
        roi_specified_dataset_item = self.dataset_item()
        assert roi_specified_dataset_item.height == 8

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_item_get_annotations(self):
        """
        <b>Description:</b>
        Check DatasetItemEntity class "get_annotations" method

        <b>Input data:</b>
        DatasetItemEntity class object with specified "media", "annotation_scene", "roi", "metadata" and "subset"
        parameters

        <b>Expected results:</b>
        Test passes if list returned by "get_annotations" method is equal to expected

        <b>Steps</b>
        1. Check list returned by "get_annotations" method with "include_empty" parameter is "False" and "labels"
        parameter is "None" for full-box ROI DatasetItemEntity
        2. Check list returned by "get_annotations" method for not full-box ROI DatasetItemEntity
        3. Check list returned by "get_annotations" method with specified "labels" parameter for full-box ROI
        DatasetItemEntity
        4. Check list returned by "get_annotations" method with "include_empty" parameter is "True" for full-box ROI
        DatasetItemEntity
        """
        # Checking list returned by "get_annotations" method with "include_empty" parameter is "False" and "labels"
        # parameter is "None" for full-box ROI DatasetItemEntity
        full_box_roi_dataset_item = self.default_values_dataset_item()
        full_box_annotations = list(
            full_box_roi_dataset_item.annotation_scene.annotations
        )
        assert full_box_roi_dataset_item.get_annotations() == full_box_annotations
        assert (
            full_box_roi_dataset_item.get_annotations(ios_threshold=0.7)
            == full_box_annotations
        )
        # Checking list returned by "get_annotations" method for not full-box ROI DatasetItemEntity
        non_full_box_roi_dataset_item = self.dataset_item()
        non_full_box_annotations = list(
            non_full_box_roi_dataset_item.annotation_scene.annotations
        )
        expected_annotations = []
        # Creating denormalized annotations
        for annotation in non_full_box_annotations:
            labels = annotation.get_labels(include_empty=False)
            denormalized_shape = annotation.shape.denormalize_wrt_roi_shape(
                non_full_box_roi_dataset_item.roi.shape
            )
            expected_annotations.append(
                Annotation(shape=denormalized_shape, labels=labels)
            )
        # Check for "labels" is "None"
        # Check for non-specified ios_threshold parameter
        annotations_actual = non_full_box_roi_dataset_item.get_annotations()
        self.compare_annotations(annotations_actual, expected_annotations)
        # Check for specified ios_threshold parameter
        assert non_full_box_roi_dataset_item.get_annotations(ios_threshold=1.1) == []
        # Checking list returned by "get_annotations" method with "include_empty" parameter is "False" for
        # full-box ROI DatasetItemEntity with specified "labels" parameter
        expected_annotations = [
            list(full_box_roi_dataset_item.annotation_scene.annotations)[0]
        ]
        annotations_actual = full_box_roi_dataset_item.get_annotations(
            labels=[self.labels()[0]]
        )
        self.compare_annotations(annotations_actual, expected_annotations)
        # Checking list returned by "get_annotations" method with "include_empty" parameter set to "True" for
        # full-box ROI DatasetItemEntity
        expected_annotations = list(
            full_box_roi_dataset_item.annotation_scene.annotations
        )
        annotations_actual = full_box_roi_dataset_item.get_annotations(
            include_empty=True
        )
        self.compare_annotations(annotations_actual, expected_annotations)

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_item_append_annotations(self):
        """
        <b>Description:</b>
        Check DatasetItemEntity class "append_annotations" method

        <b>Input data:</b>
        DatasetItemEntity class object with specified "media", "annotation_scene", "roi", "metadata" and "subset"
        parameters

        <b>Expected results:</b>
        Test passes if annotations list returned after "append_annotations" method is equal to expected

        <b>Steps</b>
        1. Check annotations list returned after "append_annotations" method with specified non-included annotations
        2. Check annotations list returned after "append_annotations" method with incorrect shape annotation
        """
        # Checking annotations list returned after "append_annotations" method with specified non-included annotations
        dataset_item = self.default_values_dataset_item()
        full_box_annotations = list(dataset_item.annotation_scene.annotations)
        annotations_to_add = self.annotations_to_add()
        dataset_item.append_annotations(annotations_to_add)
        assert (
            dataset_item.annotation_scene.annotations
            == full_box_annotations + annotations_to_add
        )
        # Checking annotations list returned after "append_annotations" method with incorrect shape annotation
        dataset_item = self.default_values_dataset_item()
        incorrect_shape_label = LabelEntity(
            name="Label for incorrect shape",
            domain=Domain.CLASSIFICATION,
            color=Color(red=80, green=70, blue=155),
            id=ID("incorrect_shape_label"),
        )
        incorrect_shape_scored_label = ScoredLabel(incorrect_shape_label)
        incorrect_polygon = Polygon(
            [Point(x=0.01, y=0.1), Point(x=0.35, y=0.1), Point(x=0.35, y=0.1)]
        )
        incorrect_shape_annotation = Annotation(
            shape=incorrect_polygon,
            labels=[incorrect_shape_scored_label],
            id=ID("incorrect_shape_annotation"),
        )
        dataset_item.append_annotations(
            annotations_to_add + [incorrect_shape_annotation]
        )
        assert (
            dataset_item.annotation_scene.annotations
            == full_box_annotations + annotations_to_add
        )

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_item_get_roi_labels(self):
        """
        <b>Description:</b>
        Check DatasetItemEntity class "get_roi_labels" method

        <b>Input data:</b>
        DatasetItemEntity class object with specified "media", "annotation_scene", "roi", "metadata" and "subset"
        parameters

        <b>Expected results:</b>
        Test passes if annotations list returned by "get_roi_labels" method is equal to expected

        <b>Steps</b>
        1. Check annotations list returned by "get_roi_labels" for non-specified "labels" parameter
        2. Check annotations list returned by "get_roi_labels" for specified "labels" parameter
        """
        dataset_item = self.dataset_item()
        roi_labels = self.roi_labels()
        # Checking annotations list returned by "get_roi_labels" method with non-specified labels parameter
        # Scenario for "include_empty" is "False"
        assert dataset_item.get_roi_labels() == [roi_labels[0]]
        # Scenario for "include_empty" is "True"
        assert dataset_item.get_roi_labels(include_empty=True) == roi_labels
        # Checking annotations list returned by "get_roi_labels" method with specified labels parameter
        empty_roi_label = roi_labels[1]
        # Scenario for "include_empty" is "False"
        assert dataset_item.get_roi_labels(labels=[empty_roi_label]) == []
        # Scenario for "include_empty" is "True"
        assert dataset_item.get_roi_labels([empty_roi_label], True) == [empty_roi_label]

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_item_get_shapes_labels(self):
        """
        <b>Description:</b>
        Check DatasetItemEntity class "get_shapes_labels" method

        <b>Input data:</b>
        DatasetItemEntity class object with specified "media", "annotation_scene", "roi", "metadata" and "subset"
        parameters

        <b>Expected results:</b>
        Test passes if annotations list returned by "get_roi_labels" method is equal to expected

        <b>Steps</b>
        1. Check annotations list returned by get_shapes_labels for non-specified "labels" parameter
        2. Check annotations list returned by get_shapes_labels for specified "labels" parameter
        """
        dataset_item = self.default_values_dataset_item()
        labels = self.labels()
        detection_label = labels[0]
        segmentation_label = labels[1]
        # Checking annotations list returned by "get_shapes_labels" method with non-specified "labels" parameter
        # Scenario for "include_empty" is "False"
        assert dataset_item.get_shapes_labels() == [detection_label]
        # Scenario for "include_empty" is "True"
        shapes_labels_actual = dataset_item.get_shapes_labels(include_empty=True)
        assert len(shapes_labels_actual) == 2
        assert detection_label in shapes_labels_actual
        assert segmentation_label in shapes_labels_actual
        # Checking annotations list returned by "get_shapes_labels" method with specified "labels" parameter
        # Scenario for "include_empty" is "False"
        non_included_label = LabelEntity("Non-included label", Domain.CLASSIFICATION)
        list_labels = [segmentation_label, non_included_label]
        assert dataset_item.get_shapes_labels(labels=list_labels) == []
        # Scenario for "include_empty" is "True", expected that non_included label will not be shown
        assert dataset_item.get_shapes_labels(list_labels, True) == [segmentation_label]

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_item_append_labels(self):
        """
        <b>Description:</b>
        Check DatasetItemEntity class "append_labels" method

        <b>Input data:</b>
        DatasetItemEntity class object with specified "media", "annotation_scene", "roi", "metadata" and "subset"
        parameters

        <b>Expected results:</b>
        Test passes if annotations list returned returned after using "append_labels" method is equal to expected

        <b>Steps</b>
        1. Check annotations list after "append_labels" method for DatasetItemEntity object with ROI-annotation
        specified in annotation_scene.annotations
        2. Check annotations list after "append_labels" method for DatasetItemEntity object with non-specified
        ROI-annotation in annotation_scene.annotations
        """
        annotation_labels = self.labels()
        labels_to_add = self.labels_to_add()
        scored_labels_to_add = [
            ScoredLabel(labels_to_add[0]),
            ScoredLabel(labels_to_add[1]),
        ]
        media = self.generate_random_image()
        roi_labels = self.roi_labels()
        roi_scored_labels = self.roi_scored_labels()
        roi = self.roi()
        equal_roi = self.roi()
        annotations = self.annotations()
        annotations_with_roi = annotations + [equal_roi]
        annotations_scene = AnnotationSceneEntity(
            annotations_with_roi, AnnotationSceneKind.ANNOTATION
        )
        # Scenario for checking "append_labels" method for DatasetItemEntity object with ROI-annotation specified in
        # annotation_scene.annotations object
        roi_label_dataset_item = DatasetItemEntity(media, annotations_scene, roi)
        roi_label_dataset_item.append_labels(scored_labels_to_add)
        # Check for include_empty is "False"
        expected_labels = [annotation_labels[0], roi_labels[0], labels_to_add[0]]
        assert roi_label_dataset_item.annotation_scene.get_labels() == expected_labels
        expected_labels = [roi_scored_labels[0], scored_labels_to_add[0]]
        self.check_roi_equal_annotation(roi_label_dataset_item, expected_labels)
        # Check for include_empty is "True"
        expected_labels = annotation_labels + roi_labels + labels_to_add
        assert (
            roi_label_dataset_item.annotation_scene.get_labels(True) == expected_labels
        )
        expected_labels = roi_scored_labels + scored_labels_to_add
        self.check_roi_equal_annotation(roi_label_dataset_item, expected_labels, True)
        # Scenario for checking "append_labels" method for DatasetItemEntity object with non-specified ROI-annotation in
        # annotation_scene.annotations object
        non_roi_dataset_item = self.dataset_item()
        non_roi_dataset_item.append_labels(scored_labels_to_add)
        # Check for "include_empty" is "False"
        expected_labels = [annotation_labels[0], roi_labels[0], labels_to_add[0]]
        assert non_roi_dataset_item.annotation_scene.get_labels() == expected_labels
        expected_labels = [roi_scored_labels[0], scored_labels_to_add[0]]
        self.check_roi_equal_annotation(non_roi_dataset_item, expected_labels)
        # Check for "include_empty" is "True"
        expected_labels = annotation_labels + roi_labels + labels_to_add
        assert non_roi_dataset_item.annotation_scene.get_labels(True) == expected_labels
        expected_labels = roi_scored_labels + scored_labels_to_add
        self.check_roi_equal_annotation(non_roi_dataset_item, expected_labels, True)
        # Scenario for "labels" parameter is equal to []
        dataset_item = self.dataset_item()
        dataset_item.append_labels([])
        assert dataset_item.annotation_scene.get_labels() == [annotation_labels[0]]
        assert (
            dataset_item.annotation_scene.get_labels(include_empty=True)
            == annotation_labels
        )

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_item_eq(self):
        """
        <b>Description:</b>
        Check DatasetItemEntity class __eq__ method

        <b>Input data:</b>
        DatasetItemEntity class objects with specified "media", "annotation_scene", "roi", "metadata" and "subset"
        parameters

        <b>Expected results:</b>
        Test passes if value returned by __eq__ method is equal to expected

        <b>Steps</b>
        1. Check value returned by __eq__ method for equal DatasetItemEntity objects
        2. Check value returned by __eq__ method for DatasetItemEntity objects with unequal "media", "annotation_scene",
        "roi" or "subset"  parameters
        3. Check value returned by __eq__ method for DatasetItemEntity objects with unequal "metadata" parameters
        4. Check value returned by __eq__ method for DatasetItemEntity object compared to different type object
        """
        media = self.generate_random_image()
        annotation_scene = self.annotations_entity()
        roi = self.roi()
        metadata = self.metadata()
        dataset_parameters = {
            "media": media,
            "annotation_scene": annotation_scene,
            "roi": roi,
            "metadata": metadata,
            "subset": Subset.TESTING,
        }
        dataset_item = DatasetItemEntity(**dataset_parameters)
        # Checking value returned by __eq__ method for equal DatasetItemEntity objects
        equal_dataset_item = DatasetItemEntity(**dataset_parameters)
        assert dataset_item == equal_dataset_item
        # Checking inequality of DatasetItemEntity objects with unequal initialization parameters
        unequal_annotation_scene = self.annotations_entity()
        unequal_annotation_scene.annotations.pop(0)
        unequal_values = [
            ("media", self.generate_random_image()),
            ("annotation_scene", unequal_annotation_scene),
            ("roi", None),
            ("subset", Subset.VALIDATION),
        ]
        for key, value in unequal_values:
            unequal_parameters = dict(dataset_parameters)
            unequal_parameters[key] = value
            unequal_dataset_item = DatasetItemEntity(**unequal_parameters)
            assert dataset_item != unequal_dataset_item, (
                f"Expected False returned for DatasetItemEntity objects with "
                f"unequal {key} parameters"
            )
        # Checking value returned by __eq__ method for DatasetItemEntity objects with unequal "metadata" parameters
        # expected equality
        unequal_metadata_parameters = dict(dataset_parameters)
        unequal_metadata_parameters["metadata"] = None
        unequal_metadata_dataset_item = DatasetItemEntity(**unequal_metadata_parameters)
        assert dataset_item == unequal_metadata_dataset_item
        # Checking value returned by __eq__ method for DatasetItemEntity object compared to different type object
        assert not dataset_item == str

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_item_deepcopy(self):
        """
        <b>Description:</b>
        Check DatasetItemEntity class __deepcopy__ method

        <b>Input data:</b>
        DatasetItemEntity class objects with specified "media", "annotation_scene", "roi", "metadata" and "subset"
        parameters

        <b>Expected results:</b>
        Test passes if DatasetItemEntity object created by __deepcopy__ method is equal to expected
        """
        dataset_item = self.dataset_item()
        copy_dataset = deepcopy(dataset_item)
        assert (
            dataset_item._DatasetItemEntity__roi_lock
            != copy_dataset._DatasetItemEntity__roi_lock
        )
        assert (dataset_item.media.numpy == copy_dataset.media.numpy).all()
        assert (
            dataset_item.annotation_scene.annotations
            == copy_dataset.annotation_scene.annotations
        )
        assert (
            dataset_item.annotation_scene.creation_date
            == copy_dataset.annotation_scene.creation_date
        )
        assert (
            dataset_item.annotation_scene.editor_name
            == copy_dataset.annotation_scene.editor_name
        )
        assert dataset_item.annotation_scene.id == copy_dataset.annotation_scene.id
        assert dataset_item.annotation_scene.kind == copy_dataset.annotation_scene.kind
        assert (
            dataset_item.annotation_scene.shapes == copy_dataset.annotation_scene.shapes
        )
        assert dataset_item.roi == copy_dataset.roi
        assert dataset_item.metadata == copy_dataset.metadata
        assert dataset_item.subset == copy_dataset.subset

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_item_append_metadata_item(self):
        """
        <b>Description:</b>
        Check DatasetItemEntity class "append_metadata_item" method

        <b>Input data:</b>
        DatasetItemEntity class object with specified "media", "annotation_scene", "roi", "metadata" and "subset"
        parameters

        <b>Expected results:</b>
        Test passes if "metadata" attribute after "append_metadata_item" method is equal to expected

        <b>Steps</b>
        1. Check "metadata" attribute after "append_metadata_item" method with non-specified "model" parameter
        2. Check "metadata" attribute after "append_metadata_item" method with specified "model" parameter
        """
        dataset_item = self.dataset_item()
        expected_metadata = list(dataset_item.metadata)
        # Checking metadata attribute returned after "append_metadata_item" method with non-specified "model" parameter
        data_to_append = TensorEntity(
            name="appended_metadata",
            numpy=np.random.uniform(low=0.0, high=255.0, size=(10, 15, 3)),
        )
        expected_metadata.append(MetadataItemEntity(data=data_to_append))
        dataset_item.append_metadata_item(data=data_to_append)
        assert list(dataset_item.metadata) == expected_metadata
        # Checking metadata attribute returned after "append_metadata_item" method with specified "model" parameter
        metadata_item_with_model = self.metadata_item_with_model()
        data_to_append = metadata_item_with_model.data
        model_to_append = metadata_item_with_model.model
        new_metadata_item_with_model = MetadataItemEntity(
            data_to_append, model_to_append
        )
        expected_metadata.append(new_metadata_item_with_model)
        dataset_item.append_metadata_item(data_to_append, model_to_append)
        assert list(dataset_item.metadata) == expected_metadata

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_item_get_metadata_by_name_and_model(self):
        """
        <b>Description:</b>
        Check DatasetItemEntity class "get_metadata_by_name_and_model" method

        <b>Input data:</b>
        DatasetItemEntity class object with specified "media", "annotation_scene", "roi", "metadata" and "subset"
        parameters

        <b>Expected results:</b>
        Test passes if MetadataItemEntity object returned by "get_metadata_by_name_and_model" is equal to expected

        <b>Steps</b>
        1. Check value returned by "get_metadata_by_name_and_model" method for searching metadata object with "model"
        is "None"
        2. Check value returned by "get_metadata_by_name_and_model" method for searching metadata object with specified
        "model" attribute
        3. Check value returned by "get_metadata_by_name_and_model" method for searching non-existing metadata object
        """
        dataset_item = self.dataset_item()
        metadata_item_with_model = self.metadata_item_with_model()
        dataset_model = metadata_item_with_model.model
        dataset_item.append_metadata_item(metadata_item_with_model.data, dataset_model)
        dataset_metadata = list(dataset_item.metadata)
        # Checking "get_metadata_by_name_and_model" method for "model" parameter is None
        assert dataset_item.get_metadata_by_name_and_model("test_metadata", None) == [
            dataset_metadata[0]
        ]
        # Checking "get_metadata_by_name_and_model" method for specified "model" parameter
        assert dataset_item.get_metadata_by_name_and_model(
            "appended_metadata_with_model", dataset_model
        ) == [dataset_metadata[2]]
        # Checking "get_metadata_by_name_and_model" method for searching non-existing metadata
        assert (
            dataset_item.get_metadata_by_name_and_model("test_metadata", dataset_model)
            == []
        )
        assert (
            dataset_item.get_metadata_by_name_and_model(
                "appended_metadata_with_model", None
            )
            == []
        )

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_item_setters(self):
        """
        <b>Description:</b>
        Check DatasetItemEntity class setters

        <b>Input data:</b>
        DatasetItemEntity class object with specified "media", "annotation_scene", "roi", "metadata" and "subset"
        parameters

        <b>Expected results:</b>
        Test passes if assigned values of "roi", "subset" and "annotation_scene" properties are equal to expected

        <b>Steps</b>
        1. Check value returned by "roi" property after using @roi.setter
        2. Check value returned by "subset" property after using @subset.setter
        3. Check value returned by "annotation_scene" property after using @subset.annotation_scene
        """
        dataset_item = self.dataset_item()
        # Checking value returned by "roi" property after using @roi.setter
        new_roi_label = ScoredLabel(LabelEntity("new ROI label", Domain.DETECTION))
        new_dataset_roi = Annotation(
            Rectangle(x1=0.2, y1=0.2, x2=1.0, y2=1.0), [new_roi_label]
        )
        dataset_item.roi = new_dataset_roi
        assert dataset_item.roi == new_dataset_roi
        # Checking value returned by subset property after using @subset.setter
        new_subset = Subset.TRAINING
        dataset_item.subset = new_subset
        assert dataset_item.subset == new_subset
        # Checking value returned by annotation_scene property after using @subset.annotation_scene
        new_annotation_label = ScoredLabel(
            LabelEntity("new annotation label", Domain.CLASSIFICATION)
        )
        new_annotation = Annotation(
            Rectangle(x1=0.1, y1=0, x2=0.9, y2=1.0), [new_annotation_label]
        )
        new_annotation_scene = AnnotationSceneEntity(
            [new_annotation], AnnotationSceneKind.PREDICTION
        )
        dataset_item.annotation_scene = new_annotation_scene
        assert dataset_item.annotation_scene == new_annotation_scene
