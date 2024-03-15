# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import datetime
from copy import deepcopy
from typing import List, Union

import numpy as np
import pytest

from otx.api.configuration import ConfigurableParameters
from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from otx.api.entities.color import Color
from otx.api.entities.dataset_item import DatasetItemEntity, DatasetItemEntityWithID
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.id import ID
from otx.api.entities.image import Image
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.entities.metadata import MetadataItemEntity
from otx.api.entities.model import ModelConfiguration, ModelEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.ellipse import Ellipse
from otx.api.entities.shapes.polygon import Point, Polygon
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.subset import Subset
from otx.api.entities.tensor import TensorEntity
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


class DatasetItemParameters:
    @staticmethod
    def generate_random_image() -> Image:
        image = Image(data=np.random.randint(low=0, high=255, size=(10, 16, 3)))
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
            annotations=self.annotations(),
            kind=AnnotationSceneKind.ANNOTATION,
            creation_date=datetime.datetime(year=2021, month=12, day=19),
            id=ID("annotation_entity_1"),
        )

    def default_values_dataset_item(self) -> DatasetItemEntity:
        return DatasetItemEntity(self.generate_random_image(), self.annotations_entity())

    def dataset_item(self) -> DatasetItemEntity:
        return DatasetItemEntity(
            media=self.generate_random_image(),
            annotation_scene=self.annotations_entity(),
            roi=self.roi(),
            metadata=self.metadata(),
            subset=Subset.TESTING,
            ignored_labels={self.labels()[1]},
        )

    def dataset_item_with_id(self) -> DatasetItemEntityWithID:
        return DatasetItemEntityWithID(
            id_=ID("test"),
            media=self.generate_random_image(),
            annotation_scene=self.annotations_entity(),
            roi=self.roi(),
            metadata=self.metadata(),
            subset=Subset.TESTING,
            ignored_labels={self.labels()[1]},
        )


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestDatasetItemEntity:
    @staticmethod
    def compare_denormalized_annotations(actual_annotations, expected_annotations) -> None:
        assert len(actual_annotations) == len(expected_annotations)
        for index in range(len(expected_annotations)):
            actual_annotation = actual_annotations[index]
            expected_annotation = expected_annotations[index]
            # Redefining id and modification_date required because of new Annotation objects created after shape
            # denormalize
            actual_annotation.id_ = expected_annotation.id_
            actual_annotation.shape.modification_date = expected_annotation.shape.modification_date
            assert actual_annotation == expected_annotation

    @staticmethod
    def labels_to_add() -> List[LabelEntity]:
        label_to_add = LabelEntity(
            name="Label which will be added",
            domain=Domain.DETECTION,
            color=Color(red=60, green=120, blue=70),
            creation_date=datetime.datetime(year=2021, month=12, day=12),
            id=ID("label_to_add_1"),
        )
        other_label_to_add = LabelEntity(
            name="Other label to add",
            domain=Domain.SEGMENTATION,
            color=Color(red=80, green=70, blue=100),
            creation_date=datetime.datetime(year=2021, month=12, day=11),
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
            id=ID("added_annotation_2"),
        )
        return [annotation_to_add, other_annotation_to_add]

    @staticmethod
    def metadata_item_with_model() -> MetadataItemEntity:
        data = TensorEntity(
            name="appended_metadata_with_model",
            numpy=np.random.randint(low=0, high=255, size=(10, 15, 3)),
        )
        configuration = ModelConfiguration(
            configurable_parameters=ConfigurableParameters(header="Test Header"),
            label_schema=LabelSchemaEntity(),
        )
        model = ModelEntity(configuration=configuration, train_dataset=DatasetEntity())
        metadata_item_with_model = MetadataItemEntity(data=data, model=model)
        return metadata_item_with_model

    @staticmethod
    def check_roi_equal_annotation(dataset_item: DatasetItemEntity, expected_labels: list, include_empty=False) -> None:
        roi_annotation_in_scene = None
        for annotation in dataset_item.annotation_scene.annotations:
            if annotation == dataset_item.roi:
                assert annotation.get_labels(include_empty=include_empty) == expected_labels
                roi_annotation_in_scene = annotation
                break
        assert roi_annotation_in_scene

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_item_initialization(self):
        """
        <b>Description:</b>
        Check DatasetItemEntity class object initialization

        <b>Input data:</b>
        DatasetItemEntity class objects with specified "media", "annotation_scene", "roi", "metadata" and "subset"
        parameters

        <b>Expected results:</b>
        Test passes if attributes of DatasetItemEntity class object are equal to expected

        <b>Steps</b>
        1. Check attributes of DatasetItemEntity object initialized with default optional parameters
        2. Check attributes of DatasetItemEntity object initialized with specified optional parameters
        """
        media = DatasetItemParameters.generate_random_image()
        annotations_scene = DatasetItemParameters().annotations_entity()
        # Checking attributes of DatasetItemEntity object initialized with default optional parameters
        default_values_dataset_item = DatasetItemEntity(media, annotations_scene)
        assert default_values_dataset_item.media == media
        assert default_values_dataset_item.annotation_scene == annotations_scene
        assert not default_values_dataset_item.get_metadata()
        assert default_values_dataset_item.subset == Subset.NONE
        assert default_values_dataset_item.ignored_labels == set()
        # Checking attributes of DatasetItemEntity object initialized with specified optional parameters
        roi = DatasetItemParameters().roi()
        metadata = DatasetItemParameters.metadata
        subset = Subset.TESTING
        ignored_labels = set(DatasetItemParameters().labels())
        specified_values_dataset_item = DatasetItemEntity(
            media, annotations_scene, roi, metadata, subset, ignored_labels
        )
        assert specified_values_dataset_item.media == media
        assert specified_values_dataset_item.annotation_scene == annotations_scene
        assert specified_values_dataset_item.roi == roi
        assert specified_values_dataset_item.get_metadata() == metadata
        assert specified_values_dataset_item.subset == subset
        assert specified_values_dataset_item.ignored_labels == ignored_labels

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_item_repr(self):
        """
        <b>Description:</b>
        Check DatasetItemEntity class __repr__ method

        <b>Input data:</b>
        DatasetItemEntity class objects with specified "media", "annotation_scene", "roi", "metadata" and "subset"
        parameters

        <b>Expected results:</b>
        Test passes if value returned by __repr__ method is equal to expected

        <b>Steps</b>
        1. Check value returned by __repr__ method for  DatasetItemEntity object with default optional parameters
        2. Check value returned by __repr__ method for  DatasetItemEntity object with specified optional parameters
        """
        media = DatasetItemParameters.generate_random_image()
        annotation_scene = DatasetItemParameters().annotations_entity()
        # Checking __repr__ method for DatasetItemEntity object initialized with default optional parameters
        default_values_dataset_item = DatasetItemEntity(media, annotation_scene)
        generated_roi = default_values_dataset_item.roi

        assert repr(default_values_dataset_item) == (
            f"DatasetItemEntity(media=Image(with data, width=16, height=10), "
            f"annotation_scene={annotation_scene}, roi={generated_roi}, "
            f"subset=NONE), meta=[]"
        )
        # Checking __repr__ method for DatasetItemEntity object initialized with specified optional parameters
        roi = DatasetItemParameters().roi()
        metadata = DatasetItemParameters.metadata()
        subset = Subset.TESTING
        specified_values_dataset_item = DatasetItemEntity(media, annotation_scene, roi, metadata, subset)
        assert repr(specified_values_dataset_item) == (
            f"DatasetItemEntity(media=Image(with data, width=16, height=10), annotation_scene={annotation_scene}, "
            f"roi={roi}, subset=TESTING), meta={metadata}"
        )

    @pytest.mark.priority_medium
    @pytest.mark.unit
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
        media = DatasetItemParameters.generate_random_image()
        annotations = DatasetItemParameters().annotations()
        annotation_scene = DatasetItemParameters().annotations_entity()
        roi = DatasetItemParameters().roi()
        metadata = DatasetItemParameters.metadata()
        # Checking "roi" property for DatasetItemEntity with specified "roi" parameter
        specified_roi_dataset_item = DatasetItemParameters().dataset_item()
        assert specified_roi_dataset_item.roi == roi
        # Checking that "roi" property is equal to full_box for DatasetItemEntity with not specified "roi" parameter
        non_specified_roi_dataset_item = DatasetItemEntity(media, annotation_scene, metadata=metadata)
        default_roi = non_specified_roi_dataset_item.roi.shape
        assert isinstance(default_roi, Rectangle)
        assert Rectangle.is_full_box(default_roi)
        # Checking that "roi" property will be equal to full_box for DatasetItemEntity with not specified "roi" but one
        # of Annotation objects in annotation_scene is equal to full Rectangle
        full_box_label = LabelEntity("Full-box label", Domain.DETECTION, id=ID("full_box_label"))
        full_box_annotation = Annotation(Rectangle.generate_full_box(), [ScoredLabel(full_box_label)])
        annotations.append(full_box_annotation)
        annotation_scene.annotations.append(full_box_annotation)
        full_box_label_dataset_item = DatasetItemEntity(media, annotation_scene, metadata=metadata)
        assert full_box_label_dataset_item.roi is full_box_annotation

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_item_roi_numpy(self):
        """
        <b>Description:</b>
        Check DatasetItemEntity class "roi_numpy" method

        <b>Input data:</b>
        DatasetItemEntity class object with specified "media", "annotation_scene", "roi", "metadata" and "subset"
        parameters

        <b>Expected results:</b>
        Test passes if array returned by "roi_numpy" method is equal to expected

        <b>Steps</b>
        1. Check array returned by roi_numpy method with not specified "roi" parameter for DatasetItemEntity with
        "roi" attribute is "None"
        2. Check array returned by roi_numpy method with Rectangle-shape "roi" parameter
        3. Check array returned by roi_numpy method with Ellipse-shape "roi" parameter
        4. Check array returned by roi_numpy method with Polygon-shape "roi" parameter
        5. Check array returned by roi_numpy method with non-specified "roi" parameter for DatasetItemEntity with "roi"
        attribute
        """
        media = DatasetItemParameters.generate_random_image()
        annotation_scene = DatasetItemParameters().annotations_entity()
        roi_label = LabelEntity("ROI label", Domain.DETECTION, id=ID("roi_label"))
        dataset_item = DatasetItemEntity(media, annotation_scene)
        # Checking array returned by "roi_numpy" method with non-specified "roi" parameter for DatasetItemEntity
        # "roi" attribute is "None"
        assert np.array_equal(dataset_item.roi_numpy(), media.numpy)
        # Checking array returned by "roi_numpy" method with specified Rectangle-shape "roi" parameter
        rectangle_roi = Annotation(
            Rectangle(x1=0.2, y1=0.1, x2=0.8, y2=0.9),
            [ScoredLabel(roi_label)],
            ID("rectangle_roi"),
        )
        assert np.array_equal(dataset_item.roi_numpy(rectangle_roi), media.numpy[1:9, 3:13])
        # Checking array returned by "roi_numpy" method with specified Ellipse-shape "roi" parameter
        ellipse_roi = Annotation(
            Ellipse(x1=0.1, y1=0.0, x2=0.9, y2=0.8),
            [ScoredLabel(roi_label)],
            ID("ellipse_roi"),
        )
        assert np.array_equal(dataset_item.roi_numpy(ellipse_roi), media.numpy[0:8, 2:14])
        # Checking array returned by "roi_numpy" method with specified Polygon-shape "roi" parameter
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
            id=ID("polygon_roi"),
        )
        assert np.array_equal(dataset_item.roi_numpy(polygon_roi), media.numpy[4:8, 5:13])
        # Checking array returned by "roi_numpy" method with not specified "roi" parameter for DatasetItemEntity with
        # "roi" attribute
        roi_specified_dataset_item = DatasetItemEntity(media, annotation_scene, DatasetItemParameters().roi())
        roi_specified_dataset_item.roi_numpy()
        assert np.array_equal(roi_specified_dataset_item.roi_numpy(), media.numpy[1:9, 2:14])

    @pytest.mark.priority_medium
    @pytest.mark.unit
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
        1. Check array returned by "numpy" property for DatasetItemEntity with "roi" attribute is "None"
        2. Check array returned by "numpy" property for DatasetItemEntity with specified "roi" attribute
        """
        # Checking array returned by numpy property for DatasetItemEntity with "roi" attribute is "None"
        none_roi_dataset_item = DatasetItemParameters().default_values_dataset_item()
        assert np.array_equal(none_roi_dataset_item.numpy, none_roi_dataset_item.roi_numpy())
        # Checking array returned by numpy property for DatasetItemEntity with specified "roi" attribute
        roi_specified_dataset_item = DatasetItemParameters().dataset_item()
        assert np.array_equal(roi_specified_dataset_item.numpy, roi_specified_dataset_item.roi_numpy())

    @pytest.mark.priority_medium
    @pytest.mark.unit
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
        # Checking value returned by "width" property for DatasetItemEntity with "roi" attribute is "None"
        none_roi_dataset_item = DatasetItemParameters().default_values_dataset_item()
        assert none_roi_dataset_item.width == 16
        # Checking value returned by "width" property for DatasetItemEntity with specified "roi" attribute
        roi_specified_dataset_item = DatasetItemParameters().dataset_item()
        assert roi_specified_dataset_item.width == 12

    @pytest.mark.priority_medium
    @pytest.mark.unit
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
        # Checking value returned by "width" property for DatasetItemEntity with None "roi" attribute
        none_roi_dataset_item = DatasetItemParameters().default_values_dataset_item()
        assert none_roi_dataset_item.height == 10
        # Checking value returned by "width" property for DatasetItemEntity with specified "roi" attribute
        roi_specified_dataset_item = DatasetItemParameters().dataset_item()
        assert roi_specified_dataset_item.height == 8

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.xfail
    # TODO: Fix this - https://jira.devtools.intel.com/browse/CVS-91526
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
        1. Check that get_annotations returns all annotations in the dataset item if the ROI is a full box
        2. Check that after adding the parameter "labels", only the annotations with that label are returned
        3. Check that for a ROI that includes only one of the annotations, only that annotation is returned
        """
        # Check that get_annotations returns all items if the ROI is a full box.
        full_box_roi_dataset_item = DatasetItemParameters().default_values_dataset_item()
        full_box_annotations = list(full_box_roi_dataset_item.annotation_scene.annotations)
        result_annotations = full_box_roi_dataset_item.get_annotations(include_empty=True)
        expected_annotations = full_box_annotations
        self.compare_denormalized_annotations(result_annotations, expected_annotations)

        # Check that get_annotations returns only the items with the right label if the "labels" param is used
        first_annotation = full_box_roi_dataset_item.annotation_scene.annotations[0]
        first_annotation_label = first_annotation.get_labels()[0].label
        result_annotations = full_box_roi_dataset_item.get_annotations(
            labels=[first_annotation_label], include_empty=True
        )
        expected_annotations = [first_annotation]
        self.compare_denormalized_annotations(result_annotations, expected_annotations)

        # Check that get_annotations only returns the annotations whose center falls within the ROI
        partial_box_dataset_item = deepcopy(full_box_roi_dataset_item)
        partial_box_dataset_item.roi = Annotation(shape=Rectangle(x1=0.0, y1=0.0, x2=0.4, y2=0.5), labels=[])
        expected_annotation = deepcopy(first_annotation)
        expected_annotation.shape = expected_annotation.shape.denormalize_wrt_roi_shape(
            roi_shape=partial_box_dataset_item.roi.shape
        )
        result_annotations = partial_box_dataset_item.get_annotations(include_empty=True)
        self.compare_denormalized_annotations(result_annotations, [expected_annotation])

        # Check if ignored labels are properly removed
        ignore_labels_dataset_item = DatasetItemParameters().default_values_dataset_item()
        ignore_labels_dataset_item.ignored_labels = ignore_labels_dataset_item.get_shapes_labels(
            include_ignored=True, include_empty=True
        )
        assert ignore_labels_dataset_item.get_annotations(include_empty=True) == []

    @pytest.mark.priority_medium
    @pytest.mark.unit
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
        dataset_item = DatasetItemParameters().default_values_dataset_item()
        full_box_annotations = list(dataset_item.annotation_scene.annotations)
        annotations_to_add = self.annotations_to_add()
        normalized_annotations = []
        for annotation in annotations_to_add:
            normalized_annotations.append(
                Annotation(
                    shape=annotation.shape.normalize_wrt_roi_shape(dataset_item.roi.shape),
                    labels=annotation.get_labels(),
                )
            )
        dataset_item.append_annotations(annotations_to_add)
        # Random id is generated for normalized annotations
        normalized_annotations[0].id_ = dataset_item.annotation_scene.annotations[2].id_
        normalized_annotations[1].id_ = dataset_item.annotation_scene.annotations[3].id_
        assert dataset_item.annotation_scene.annotations == full_box_annotations + normalized_annotations
        # Checking annotations list returned after "append_annotations" method with incorrect shape annotation
        incorrect_shape_label = LabelEntity(
            name="Label for incorrect shape",
            domain=Domain.CLASSIFICATION,
            color=Color(red=80, green=70, blue=155),
            id=ID("incorrect_shape_label"),
        )
        incorrect_polygon = Polygon([Point(x=0.01, y=0.1), Point(x=0.35, y=0.1), Point(x=0.35, y=0.1)])
        incorrect_shape_annotation = Annotation(
            shape=incorrect_polygon,
            labels=[ScoredLabel(incorrect_shape_label)],
            id=ID("incorrect_shape_annotation"),
        )
        dataset_item.append_annotations([incorrect_shape_annotation])
        assert dataset_item.annotation_scene.annotations == full_box_annotations + normalized_annotations

    @pytest.mark.priority_medium
    @pytest.mark.unit
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
        3. Check annotations list returned by "get_roi_labels" if dataset item ignores a label
        """
        dataset_item = DatasetItemParameters().dataset_item()
        roi_labels = DatasetItemParameters.roi_labels()
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
        # Scenario for ignored labels
        dataset_item.ignored_labels = [empty_roi_label]
        assert dataset_item.get_roi_labels([empty_roi_label], True) == []

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_item_get_shapes_labels(self):
        """
        <b>Description:</b>
        Check DatasetItemEntity class "get_shapes_labels" method

        <b>Input data:</b>
        DatasetItemEntity class object with specified "media", "annotation_scene", "roi", "metadata" and "subset"
        parameters

        <b>Expected results:</b>
        Test passes if labels list returned by "get_shapes_labels" method is equal to expected

        <b>Steps</b>
        1. Check labels list returned by "get_shapes_labels" for non-specified "labels" parameter
        2. Check labels list returned by "get_shapes_labels" for specified "labels" parameter
        3. Check labels list returned by "get_shapes_labels" if dataset_item ignores labels
        """
        dataset_item = DatasetItemParameters().default_values_dataset_item()
        labels = DatasetItemParameters.labels()
        detection_label = labels[0]
        segmentation_label = labels[1]
        # Checking labels list returned by "get_shapes_labels" method with non-specified "labels" parameter
        # Scenario for "include_empty" is "False"
        assert dataset_item.get_shapes_labels() == [detection_label]
        # Scenario for "include_empty" is "True"
        shapes_labels_actual = dataset_item.get_shapes_labels(include_empty=True)
        assert len(shapes_labels_actual) == 2
        assert isinstance(shapes_labels_actual, list)
        assert detection_label in shapes_labels_actual
        assert segmentation_label in shapes_labels_actual
        # Checking labels list returned by "get_shapes_labels" method with specified "labels" parameter
        # Scenario for "include_empty" is "False"
        non_included_label = LabelEntity("Non-included label", Domain.CLASSIFICATION)
        list_labels = [segmentation_label, non_included_label]
        assert dataset_item.get_shapes_labels(labels=list_labels) == []
        # Scenario for "include_empty" is "True", expected that non_included label will not be shown
        assert dataset_item.get_shapes_labels(list_labels, include_empty=True) == [segmentation_label]
        # Check ignore labels functionality
        dataset_item.ignored_labels = [detection_label]
        assert dataset_item.get_shapes_labels(include_empty=True, include_ignored=False) == [segmentation_label]
        assert dataset_item.get_shapes_labels(include_empty=False, include_ignored=True) == [detection_label]

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_item_append_labels(self):
        """
        <b>Description:</b>
        Check DatasetItemEntity class "append_labels" method

        <b>Input data:</b>
        DatasetItemEntity class object with specified "media", "annotation_scene", "roi", "metadata" and "subset"
        parameters

        <b>Expected results:</b>
        Test passes if annotations list returned after using "append_labels" method is equal to expected

        <b>Steps</b>
        1. Check annotations list after "append_labels" method for DatasetItemEntity object with ROI-annotation
        specified in annotation_scene.annotations
        2. Check annotations list after "append_labels" method for DatasetItemEntity object with non-specified
        ROI-annotation in annotation_scene.annotations
        """
        annotation_labels = DatasetItemParameters.labels()
        labels_to_add = self.labels_to_add()
        scored_labels_to_add = [
            ScoredLabel(labels_to_add[0]),
            ScoredLabel(labels_to_add[1]),
        ]
        media = DatasetItemParameters.generate_random_image()
        roi_labels = DatasetItemParameters.roi_labels()
        roi_scored_labels = DatasetItemParameters().roi_scored_labels()
        roi = DatasetItemParameters().roi()
        equal_roi = DatasetItemParameters().roi()
        annotations = DatasetItemParameters().annotations()
        annotations_with_roi = annotations + [equal_roi]
        annotations_scene = AnnotationSceneEntity(annotations_with_roi, AnnotationSceneKind.ANNOTATION)
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
        assert roi_label_dataset_item.annotation_scene.get_labels(True) == expected_labels
        expected_labels = roi_scored_labels + scored_labels_to_add
        self.check_roi_equal_annotation(roi_label_dataset_item, expected_labels, True)
        # Scenario for checking "append_labels" method for DatasetItemEntity object with non-specified ROI-annotation in
        # annotation_scene.annotations object
        non_roi_dataset_item = DatasetItemParameters().dataset_item()
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
        dataset_item = DatasetItemParameters().dataset_item()
        dataset_item.append_labels([])
        assert dataset_item.annotation_scene.get_labels() == [annotation_labels[0]]
        assert dataset_item.annotation_scene.get_labels(include_empty=True) == annotation_labels

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_item_eq(self):
        """
        <b>Description:</b>
        Check DatasetItemEntity class __eq__ method

        <b>Input data:</b>
        DatasetItemEntity class objects with specified "media", "annotation_scene", "roi", "metadata", "subset"
        and "ignored_labels" parameters

        <b>Expected results:</b>
        Test passes if value returned by __eq__ method is equal to expected

        <b>Steps</b>
        1. Check value returned by __eq__ method for equal DatasetItemEntity objects
        2. Check value returned by __eq__ method for DatasetItemEntity objects with unequal "media", "annotation_scene",
        "roi", "subset" or "ignored_labels" parameters
        3. Check value returned by __eq__ method for DatasetItemEntity objects with unequal "metadata" parameters
        4. Check value returned by __eq__ method for DatasetItemEntity object compared to different type object
        """
        media = DatasetItemParameters.generate_random_image()
        annotation_scene = DatasetItemParameters().annotations_entity()
        roi = DatasetItemParameters().roi()
        metadata = DatasetItemParameters.metadata()
        ignored_labels = DatasetItemParameters.labels()[:1]
        dataset_parameters = {
            "media": media,
            "annotation_scene": annotation_scene,
            "roi": roi,
            "metadata": metadata,
            "subset": Subset.TESTING,
            "ignored_labels": ignored_labels,
        }
        dataset_item = DatasetItemEntity(**dataset_parameters)
        # Checking value returned by __eq__ method for equal DatasetItemEntity objects
        equal_dataset_item = DatasetItemEntity(**dataset_parameters)
        assert dataset_item == equal_dataset_item
        # Checking inequality of DatasetItemEntity objects with unequal initialization parameters
        unequal_annotation_scene = DatasetItemParameters().annotations_entity()
        unequal_annotation_scene.annotations.pop(0)
        unequal_ignored_labels = DatasetItemParameters.labels()[1:]
        unequal_values = [
            ("media", DatasetItemParameters.generate_random_image()),
            ("annotation_scene", unequal_annotation_scene),
            ("roi", None),
            ("subset", Subset.VALIDATION),
            ("ignored_labels", unequal_ignored_labels),
        ]
        for key, value in unequal_values:
            unequal_parameters = dict(dataset_parameters)
            unequal_parameters[key] = value
            unequal_dataset_item = DatasetItemEntity(**unequal_parameters)
            assert dataset_item != unequal_dataset_item, (
                f"Expected False returned for DatasetItemEntity objects with " f"unequal {key} parameters"
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
    @pytest.mark.unit
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
        dataset_item = DatasetItemParameters().dataset_item()
        copy_dataset = deepcopy(dataset_item)
        assert dataset_item._DatasetItemEntity__roi_lock != copy_dataset._DatasetItemEntity__roi_lock
        assert np.array_equal(dataset_item.media.numpy, copy_dataset.media.numpy)
        assert dataset_item.annotation_scene.annotations == copy_dataset.annotation_scene.annotations
        assert dataset_item.annotation_scene.creation_date == copy_dataset.annotation_scene.creation_date
        assert dataset_item.annotation_scene.editor_name == copy_dataset.annotation_scene.editor_name
        assert dataset_item.annotation_scene.id_ == copy_dataset.annotation_scene.id_
        assert dataset_item.annotation_scene.kind == copy_dataset.annotation_scene.kind
        assert dataset_item.annotation_scene.shapes == copy_dataset.annotation_scene.shapes
        assert dataset_item.roi == copy_dataset.roi
        assert dataset_item.get_metadata() == copy_dataset.get_metadata()
        assert dataset_item.subset == copy_dataset.subset

    @pytest.mark.priority_medium
    @pytest.mark.unit
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
        dataset_item = DatasetItemParameters().dataset_item()
        expected_metadata = dataset_item.get_metadata()
        # Checking metadata attribute returned after "append_metadata_item" method with non-specified "model" parameter
        data_to_append = TensorEntity(
            name="appended_metadata",
            numpy=np.random.uniform(low=0.0, high=255.0, size=(10, 15, 3)),
        )
        expected_metadata.append(MetadataItemEntity(data=data_to_append))
        dataset_item.append_metadata_item(data=data_to_append)
        assert dataset_item.get_metadata() == expected_metadata
        # Checking metadata attribute returned after "append_metadata_item" method with specified "model" parameter
        metadata_item_with_model = self.metadata_item_with_model()
        data_to_append = metadata_item_with_model.data
        model_to_append = metadata_item_with_model.model
        new_metadata_item_with_model = MetadataItemEntity(data_to_append, model_to_append)
        expected_metadata.append(new_metadata_item_with_model)
        dataset_item.append_metadata_item(data_to_append, model_to_append)
        assert dataset_item.get_metadata() == expected_metadata

    @pytest.mark.priority_medium
    @pytest.mark.unit
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
        dataset_item = DatasetItemParameters().dataset_item()
        metadata_item_with_model = self.metadata_item_with_model()
        dataset_model = metadata_item_with_model.model
        dataset_item.append_metadata_item(metadata_item_with_model.data, dataset_model)
        dataset_metadata = dataset_item.get_metadata()
        # Checking "get_metadata_by_name_and_model" method for "model" parameter is "None"
        assert dataset_item.get_metadata_by_name_and_model("test_metadata", None) == [dataset_metadata[0]]
        # Checking "get_metadata_by_name_and_model" method for specified "model" parameter
        assert dataset_item.get_metadata_by_name_and_model("appended_metadata_with_model", dataset_model) == [
            dataset_metadata[2]
        ]
        # Checking "get_metadata_by_name_and_model" method for searching non-existing metadata
        assert dataset_item.get_metadata_by_name_and_model("test_metadata", dataset_model) == []
        assert dataset_item.get_metadata_by_name_and_model("appended_metadata_with_model", None) == []

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_item_setters(self):
        """
        <b>Description:</b>
        Check DatasetItemEntity class "roi", "subset" and "annotation_scene" setters

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
        dataset_item = DatasetItemParameters().dataset_item()
        # Checking value returned by "roi" property after using @roi.setter
        new_roi_label = ScoredLabel(LabelEntity("new ROI label", Domain.DETECTION))
        new_dataset_roi = Annotation(Rectangle(x1=0.2, y1=0.2, x2=1.0, y2=1.0), [new_roi_label])
        dataset_item.roi = new_dataset_roi
        assert dataset_item.roi == new_dataset_roi
        # Checking value returned by subset property after using @subset.setter
        new_subset = Subset.TRAINING
        dataset_item.subset = new_subset
        assert dataset_item.subset == new_subset
        # Checking value returned by annotation_scene property after using @annotation_scene.setter
        new_annotation_label = ScoredLabel(LabelEntity("new annotation label", Domain.CLASSIFICATION))
        new_annotation = Annotation(Rectangle(x1=0.1, y1=0, x2=0.9, y2=1.0), [new_annotation_label])
        new_annotation_scene = AnnotationSceneEntity([new_annotation], AnnotationSceneKind.PREDICTION)
        dataset_item.annotation_scene = new_annotation_scene
        assert dataset_item.annotation_scene == new_annotation_scene

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.parametrize("func_name", ["dataset_item", "dataset_item_with_id"])
    def test_wrap(self, func_name):
        constructor = DatasetItemParameters()
        func = getattr(constructor, func_name)
        item: DatasetItemEntity = func()

        new_media = DatasetItemParameters().generate_random_image()
        assert item.media != new_media

        new_subset = Subset.PSEUDOLABELED
        assert item.subset != new_subset

        new_metadata = DatasetItemParameters().metadata()
        assert item.get_metadata() != new_metadata

        new_item = item.wrap(media=new_media, subset=new_subset, metadata=new_metadata)
        assert new_item.media == new_media
        assert new_item.subset == new_subset
        assert new_item.get_metadata() == new_metadata

        if hasattr(item, "id_"):
            new_id = ID("new_id")
            assert item.id_ != new_id
            item = item.wrap(id_=new_id)
            assert item.id_ == new_id
