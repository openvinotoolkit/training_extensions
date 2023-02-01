# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from unittest.mock import patch

import numpy as np
import pytest
from openvino.model_zoo.model_api.models.utils import Detection

from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from otx.api.entities.id import ID
from otx.api.entities.label import Color, Domain, LabelEntity
from otx.api.entities.label_schema import LabelGroup, LabelSchemaEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.polygon import Point, Polygon
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.usecases.exportable_code.prediction_to_annotation_converter import (
    AnomalyClassificationToAnnotationConverter,
    AnomalyDetectionToAnnotationConverter,
    AnomalySegmentationToAnnotationConverter,
    ClassificationToAnnotationConverter,
    DetectionBoxToAnnotationConverter,
    DetectionToAnnotationConverter,
    IPredictionToAnnotationConverter,
    SegmentationToAnnotationConverter,
    create_converter,
)
from otx.api.utils.time_utils import now
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestDetectionToAnnotationConverter:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_detection_to_annotation_convert(self):
        """
        <b>Description:</b>
        Check that DetectionToAnnotationConverter correctly converts Network output to list of Annotation

        <b>Input data:</b>
        Array of network output with shape [4,6]

        <b>Expected results:</b>
        Test passes if each Converted annotation has the same  values as the network output

        <b>Steps</b>
        1. Create mock network output
        2. Convert network output to Annotation
        3. Check Annotations
        """
        test_boxes = np.array(
            (
                (0, 0.6, 0.1, 0.1, 0.2, 0.3),
                (1, 0.2, 0.2, 0.1, 0.3, 0.4),
                (1, 0.7, 0.3, 0.2, 0.5, 0.6),
                (0, 0.1, 0.1, 0.1, 0.2, 0.3),
            )
        )

        labels = [
            LabelEntity("Zero", domain=Domain.DETECTION),
            LabelEntity("One", domain=Domain.DETECTION),
        ]

        converter = DetectionToAnnotationConverter(labels)

        annotation_scene = converter.convert_to_annotation(test_boxes)

        for i, annotation in enumerate(annotation_scene.annotations):
            label: ScoredLabel = next(iter(annotation.get_labels()))
            test_label = labels[int(test_boxes[i][0])]
            assert test_label.name == label.name

            assert test_boxes[i][1], label.probability

            assert test_boxes[i][2] == annotation.shape.x1
            assert test_boxes[i][3] == annotation.shape.y1
            assert test_boxes[i][4] == annotation.shape.x2
            assert test_boxes[i][5] == annotation.shape.y2

        annotation_scene = converter.convert_to_annotation(np.ndarray((0, 6)))
        assert 0 == len(annotation_scene.shapes)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_detection_to_annotation_convert_openvino_shape(self):
        """
        <b>Description:</b>
        Check that DetectionToAnnotationConverter correctly converts OpenVINO Network output to annotations

        <b>Input data:</b>
        Array of network output with shape [4,7]

        <b>Expected results:</b>
        Test passes if each Converted annotation has the same values as the network output

        <b>Steps</b>
        1. Create mock network output
        2. Convert network output to Annotation
        3. Check Annotations
        """
        test_boxes = np.array(
            (
                (-12, 0, 0.6, 0.1, 0.1, 0.2, 0.3),
                (12, 1, 0.2, 0.0, 0.1, 0.1, 0.2),
                (1234, 1, 0.7, 0.2, 0.4, 0.7, 0.5),
                (1251, 0, 0.1, 0.1, 0.1, 0.2, 0.3),
            )
        )

        labels = [
            LabelEntity("Zero", domain=Domain.DETECTION),
            LabelEntity("One", domain=Domain.DETECTION),
        ]

        converter = DetectionToAnnotationConverter(labels)

        annotation_scene = converter.convert_to_annotation(test_boxes)

        for i, annotation in enumerate(annotation_scene.annotations):
            label: ScoredLabel = next(iter(annotation.get_labels()))
            test_label = labels[int(test_boxes[i][1])]
            assert test_label.name == label.name

            assert test_boxes[i][2] == label.probability

            assert test_boxes[i][3] == annotation.shape.x1
            assert test_boxes[i][4] == annotation.shape.y1
            assert test_boxes[i][5] == annotation.shape.x2
            assert test_boxes[i][6] == annotation.shape.y2

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_detection_to_annotation_convert_invalid_input(self):
        """
        <b>Description:</b>
        Check that DetectionToAnnotationConverter raises an error if invalid inputs are provided

        <b>Input data:</b>
        Array of size [1203, 5]
        Array of size [3, 8]

        <b>Expected results:</b>
        Test passes a ValueError is raised for both inputs

        <b>Steps</b>
        1. Create DetectionToAnnotationConverter
        2. Attempt to convert array of [1203,5] to annotations
        3. Attempt to convert array of [3, 8] to annotations
        """
        labels = [
            LabelEntity("Zero", domain=Domain.DETECTION),
            LabelEntity("One", domain=Domain.DETECTION),
        ]
        converter = DetectionToAnnotationConverter(labels)

        with pytest.raises(ValueError):
            converter.convert_to_annotation(np.ndarray((1203, 5)))

        with pytest.raises(ValueError):
            converter.convert_to_annotation(np.ndarray((3, 8)))


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestIPredictionToAnnotation:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    @patch(
        (
            "otx.api.usecases.exportable_code.prediction_to_annotation_converter."
            "IPredictionToAnnotationConverter.__abstractmethods__"
        ),
        set(),
    )
    def test_i_prediction_to_annotation(self):
        """
        <b>Description:</b>
        Check "IPredictionToAnnotationConverter" class "convert_to_annotation" method

        <b>Input data:</b>
        "IPredictionToAnnotationConverter" class object, "predictions" array, "metadata" dictionary

        <b>Expected results:</b>
        Test passes if "NotImplementedError" exception is raised by "convert_to_annotation" method
        """
        i_prediction_to_annotation_converter = IPredictionToAnnotationConverter()
        with pytest.raises(NotImplementedError):
            i_prediction_to_annotation_converter.convert_to_annotation(
                predictions=np.random.randint(low=0, high=255, size=(5, 5, 3)),
                metadata={},
            )


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestCreateConverter:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_create_converter(self):
        """
        <b>Description:</b>
        Check "create_converter" function

        <b>Input data:</b>
        "converter_type" Domain-class object, "labels" LabelSchemaEntity-class object

        <b>Expected results:</b>
        Test passes if "IPredictionToAnnotationConverter" object returned by "create_converter" function is equal
        to expected

        <b>Steps</b>
        1. Check "DetectionBoxToAnnotationConverter" object returned by "create_converter" function when
        "DETECTION" domain is specified as "converter_type" parameter
        2. Check "SegmentationToAnnotationConverter" object returned by "create_converter" function when
        "SEGMENTATION" domain is specified as "converter_type" parameter
        3. Check "ClassificationToAnnotationConverter" object returned by "create_converter" function when
        "CLASSIFICATION" domain is specified as "converter_type" parameter
        4. Check "AnomalyClassificationToAnnotationConverter" object returned by "create_converter" function when
        "ANOMALY_CLASSIFICATION" domain is specified as "converter_type" parameter
        5. Check that "ValueError" exception is raised when "ANOMALY_DETECTION" or "ANOMALY_SEGMENTATION" domain is
        specified as "converter_type" parameter
        """
        # Checking "DetectionBoxToAnnotationConverter" returned by "create_converter" function when "DETECTION" is
        # specified as "converter_type"
        labels = [
            LabelEntity(name="Detection label", domain=Domain.DETECTION, id=ID("1")),
            LabelEntity(name="Other Detection label", domain=Domain.DETECTION, id=ID("2")),
        ]
        label_group = LabelGroup(name="Detection labels group", labels=labels)
        label_schema = LabelSchemaEntity(label_groups=[label_group])
        converter = create_converter(converter_type=Domain.DETECTION, labels=label_schema)
        assert isinstance(converter, DetectionToAnnotationConverter)
        assert converter.labels == labels
        # Checking "SegmentationToAnnotationConverter" returned by "create_converter" function when "SEGMENTATION"is
        # specified as "converter_type"
        labels = [
            LabelEntity(name="Segmentation label", domain=Domain.SEGMENTATION, id=ID("1")),
            LabelEntity(
                name="Other Segmentation label",
                domain=Domain.SEGMENTATION,
                id=ID("2"),
            ),
        ]
        label_group = LabelGroup(name="Segmentation labels group", labels=labels)
        label_schema = LabelSchemaEntity(label_groups=[label_group])
        converter = create_converter(converter_type=Domain.SEGMENTATION, labels=label_schema)
        assert isinstance(converter, SegmentationToAnnotationConverter)
        assert converter.label_map == {1: labels[0], 2: labels[1]}
        # Checking "ClassificationToAnnotationConverter" returned by "create_converter" function when
        # "CLASSIFICATION" is specified as "converter_type"
        labels = [
            LabelEntity(
                name="Classification label",
                domain=Domain.CLASSIFICATION,
                id=ID("1"),
            ),
            LabelEntity(
                name="Other Classification label",
                domain=Domain.CLASSIFICATION,
                id=ID("2"),
            ),
        ]
        label_group = LabelGroup(name="Classification labels group", labels=labels)
        label_schema = LabelSchemaEntity(label_groups=[label_group])
        converter = create_converter(converter_type=Domain.CLASSIFICATION, labels=label_schema)
        assert isinstance(converter, ClassificationToAnnotationConverter)
        assert converter.labels == labels
        # Checking that "AnomalyClassificationToAnnotationConverter" returned by "create_converter" function when
        # "ANOMALY_CLASSIFICATION" is specified as "converter_type"
        labels = [
            LabelEntity(name="Normal", domain=Domain.ANOMALY_CLASSIFICATION, id=ID("1")),
            LabelEntity(
                name="Anomalous",
                domain=Domain.ANOMALY_CLASSIFICATION,
                id=ID("2"),
                is_anomalous=True,
            ),
        ]
        label_group = LabelGroup(name="Anomaly classification labels group", labels=labels)
        label_schema = LabelSchemaEntity(label_groups=[label_group])
        converter = create_converter(converter_type=Domain.ANOMALY_CLASSIFICATION, labels=label_schema)
        assert isinstance(converter, AnomalyClassificationToAnnotationConverter)
        assert converter.normal_label == labels[0]
        assert converter.anomalous_label == labels[1]
        # Checking that "AnomalyDetectionToAnnotationConverter" returned by "create_converter" function when
        # "ANOMALY_DETECTION" is specified as "converter_type"
        labels = [
            LabelEntity(name="Normal", domain=Domain.ANOMALY_DETECTION, id=ID("1")),
            LabelEntity(
                name="Anomalous",
                domain=Domain.ANOMALY_DETECTION,
                id=ID("2"),
                is_anomalous=True,
            ),
        ]
        label_group = LabelGroup(name="Anomaly detection labels group", labels=labels)
        label_schema = LabelSchemaEntity(label_groups=[label_group])
        converter = create_converter(converter_type=Domain.ANOMALY_DETECTION, labels=label_schema)
        assert isinstance(converter, AnomalyDetectionToAnnotationConverter)
        assert converter.normal_label == labels[0]
        assert converter.anomalous_label == labels[1]
        # Checking that "AnomalySegmentationToAnnotationConverter" returned by "create_converter" function when
        # "ANOMALY_SEGMENTATION" is specified as "converter_type"
        labels = [
            LabelEntity(name="Normal", domain=Domain.ANOMALY_SEGMENTATION, id=ID("1")),
            LabelEntity(
                name="Anomalous",
                domain=Domain.ANOMALY_SEGMENTATION,
                id=ID("2"),
                is_anomalous=True,
            ),
        ]
        label_group = LabelGroup(name="Anomaly detection labels group", labels=labels)
        label_schema = LabelSchemaEntity(label_groups=[label_group])
        converter = create_converter(converter_type=Domain.ANOMALY_SEGMENTATION, labels=label_schema)
        assert isinstance(converter, AnomalySegmentationToAnnotationConverter)
        assert converter.normal_label == labels[0]
        assert converter.anomalous_label == labels[1]


def check_annotation_scene(annotation_scene: AnnotationSceneEntity, expected_length: int):
    assert isinstance(annotation_scene, AnnotationSceneEntity)
    assert annotation_scene.kind == AnnotationSceneKind.PREDICTION
    assert len(annotation_scene.annotations) == expected_length


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestDetectionBoxToAnnotation:
    color = Color(red=180, green=230, blue=30)
    creation_date = now()
    labels = [
        LabelEntity(
            name="Detection label",
            domain=Domain.DETECTION,
            color=color,
            creation_date=creation_date,
            id=ID("1"),
        ),
        LabelEntity(
            name="Other Detection label",
            domain=Domain.DETECTION,
            color=color,
            creation_date=creation_date,
            id=ID("2"),
        ),
    ]

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_detection_box_to_annotation_init(self):
        """
        <b>Description:</b>
        Check "DetectionBoxToAnnotationConverter" class object initialization

        <b>Input data:</b>
        "DetectionBoxToAnnotationConverter" class object with specified "labels" parameter

        <b>Expected results:</b>
        Test passes if attributes of initialized "DetectionBoxToAnnotationConverter" object are equal to expected

        <b>Steps</b>
        1. Check "labels" attribute of "DetectionBoxToAnnotationConverter" object initialized with non-empty labels
        list
        2. Check "labels" attribute of "DetectionBoxToAnnotationConverter" object initialized with empty labels list
        3. Check "labels" attributes of "DetectionBoxToAnnotationConverter" object initialized with non-empty and
        empty labels list
        """
        # Checking "labels" of "DetectionBoxToAnnotationConverter" initialized with non-empty labels list
        non_empty_labels = self.labels
        label_group = LabelGroup(name="Detection labels group", labels=non_empty_labels)
        label_schema = LabelSchemaEntity(label_groups=[label_group])
        converter = DetectionBoxToAnnotationConverter(labels=label_schema)
        assert converter.labels == non_empty_labels
        # Checking "labels" of "DetectionBoxToAnnotationConverter" initialized with empty labels list
        empty_labels = [
            LabelEntity(
                name="empty label",
                domain=Domain.DETECTION,
                is_empty=True,
                id=ID("3"),
            ),
            LabelEntity(
                name="other empty label",
                domain=Domain.DETECTION,
                is_empty=True,
                id=ID("4"),
            ),
        ]
        label_group = LabelGroup(name="Detection labels group", labels=empty_labels)
        label_schema = LabelSchemaEntity(label_groups=[label_group])
        converter = DetectionBoxToAnnotationConverter(labels=label_schema)
        assert converter.labels == []
        # Checking "labels" of "DetectionBoxToAnnotationConverter" initialized with non-empty and empty labels list
        label_group = LabelGroup(name="Detection labels group", labels=non_empty_labels + empty_labels)
        label_schema = LabelSchemaEntity(label_groups=[label_group])
        converter = DetectionBoxToAnnotationConverter(labels=label_schema)
        assert converter.labels == non_empty_labels

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_detection_box_to_annotation_convert(self):
        """
        <b>Description:</b>
        Check "DetectionBoxToAnnotationConverter" class "convert_to_annotation" method

        <b>Input data:</b>
        "DetectionBoxToAnnotationConverter" class object, "predictions" list with Detection-class objects,
        "metadata" dictionary

        <b>Expected results:</b>
        Test passes if "AnnotationSceneEntity" object returned by "convert_to_annotation" method is equal to
        expected
        """

        def check_annotation(
            actual_annotation: Annotation,
            expected_label: LabelEntity,
            expected_probability: float,
            expected_x1: float,
            expected_y1: float,
            expected_x2: float,
            expected_y2: float,
        ):
            assert isinstance(actual_annotation, Annotation)
            assert actual_annotation.get_labels() == [
                ScoredLabel(label=expected_label, probability=expected_probability)
            ]
            assert isinstance(actual_annotation.shape, Rectangle)
            assert actual_annotation.shape.x1 == pytest.approx(expected_x1)
            assert actual_annotation.shape.y1 == pytest.approx(expected_y1)
            assert actual_annotation.shape.x2 == pytest.approx(expected_x2)
            assert actual_annotation.shape.y2 == pytest.approx(expected_y2)

        labels = self.labels
        label_group = LabelGroup(name="Detection labels group", labels=labels)
        label_schema = LabelSchemaEntity(label_groups=[label_group])
        converter = DetectionBoxToAnnotationConverter(labels=label_schema)
        metadata = {
            "non-required key": 1,
            "other non-required key": 2,
            "original_shape": [10, 20],
        }
        box_1 = Detection(xmin=2, ymin=2, xmax=4, ymax=6, score=0.8, id=0)
        box_2 = Detection(xmin=6, ymin=4, xmax=10, ymax=9, score=0.9, id=1)
        predictions = [box_1, box_2]
        predictions_to_annotations = converter.convert_to_annotation(predictions=predictions, metadata=metadata)
        check_annotation_scene(annotation_scene=predictions_to_annotations, expected_length=2)
        check_annotation(
            actual_annotation=predictions_to_annotations.annotations[0],
            expected_label=labels[0],
            expected_probability=0.8,
            expected_x1=0.1,
            expected_y1=0.2,
            expected_x2=0.2,
            expected_y2=0.6,
        )
        check_annotation(
            actual_annotation=predictions_to_annotations.annotations[1],
            expected_label=labels[1],
            expected_probability=0.9,
            expected_x1=0.3,
            expected_y1=0.4,
            expected_x2=0.5,
            expected_y2=0.9,
        )


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestSegmentationToAnnotation:
    color = Color(red=180, green=230, blue=30)
    creation_date = now()
    labels = [
        LabelEntity(
            name="Segmentation label",
            domain=Domain.SEGMENTATION,
            color=color,
            creation_date=creation_date,
            id=ID("0"),
        ),
        LabelEntity(
            name="Other Segmentation label",
            domain=Domain.SEGMENTATION,
            color=color,
            creation_date=creation_date,
            id=ID("1"),
        ),
    ]

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_segmentation_to_annotation_init(self):
        """
        <b>Description:</b>
        Check "SegmentationToAnnotationConverter" class object initialization

        <b>Input data:</b>
        "SegmentationToAnnotationConverter" class object with specified "label_schema" parameter

        <b>Expected results:</b>
        Test passes if attributes of initialized "SegmentationToAnnotationConverter" object are equal to expected

        <b>Steps</b>
        1. Check "label_map" attribute of "SegmentationToAnnotationConverter" object initialized with non-empty
        labels list
        2. Check "label_map" attribute of "SegmentationToAnnotationConverter" object initialized with empty labels
        list
        3. Check "label_map" attributes of "SegmentationToAnnotationConverter" object initialized with non-empty and
        empty labels list
        """
        # Checking "label_map" of "SegmentationToAnnotationConverter" initialized with non-empty labels list
        non_empty_labels = self.labels
        label_group = LabelGroup(name="Segmentation labels group", labels=non_empty_labels)
        label_schema = LabelSchemaEntity(label_groups=[label_group])
        converter = SegmentationToAnnotationConverter(label_schema=label_schema)
        expected_non_empty_labels_map = {
            1: non_empty_labels[0],
            2: non_empty_labels[1],
        }
        assert converter.label_map == expected_non_empty_labels_map
        # Checking "label_map" of "SegmentationToAnnotationConverter" initialized with empty labels list
        empty_labels = [
            LabelEntity(
                name="empty label",
                domain=Domain.SEGMENTATION,
                is_empty=True,
                id=ID("3"),
            ),
            LabelEntity(
                name="other empty label",
                domain=Domain.SEGMENTATION,
                is_empty=True,
                id=ID("4"),
            ),
        ]
        label_group = LabelGroup(name="Segmentation labels group", labels=empty_labels)
        label_schema = LabelSchemaEntity(label_groups=[label_group])
        converter = SegmentationToAnnotationConverter(label_schema=label_schema)
        assert converter.label_map == {}
        # Checking "label_map" of "SegmentationToAnnotationConverter" initialized with non-empty and empty labels list
        label_group = LabelGroup(name="Segmentation labels group", labels=non_empty_labels + empty_labels)
        label_schema = LabelSchemaEntity(label_groups=[label_group])
        converter = SegmentationToAnnotationConverter(label_schema=label_schema)
        assert converter.label_map == expected_non_empty_labels_map

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_segmentation_to_annotation_convert(self):
        """
        <b>Description:</b>
        Check "SegmentationToAnnotationConverter" class "convert_to_annotation" method

        <b>Input data:</b>
        "SegmentationToAnnotationConverter" class object, "predictions" array with hard predictions,
        "metadata" dictionary

        <b>Expected results:</b>
        Test passes if "AnnotationSceneEntity" object returned by "convert_to_annotation" method is equal to
        expected
        """

        def check_annotation(
            actual_annotation: Annotation,
            expected_label: LabelEntity,
            expected_probability: float,
            expected_points: list,
        ):
            assert isinstance(actual_annotation, Annotation)
            annotation_labels = actual_annotation.get_labels()
            # Checking Annotation ScoredLabel
            assert len(annotation_labels) == 1
            assert isinstance(annotation_labels[0], ScoredLabel)
            assert annotation_labels[0].label == expected_label
            assert annotation_labels[0].probability == pytest.approx(expected_probability)
            # Checking Annotation Shape
            assert isinstance(actual_annotation.shape, Polygon)
            assert actual_annotation.shape.points == expected_points

        labels = self.labels
        label_group = LabelGroup(name="Segmentation labels group", labels=labels)
        label_schema = LabelSchemaEntity(label_groups=[label_group])
        converter = SegmentationToAnnotationConverter(label_schema=label_schema)
        soft_predictions = np.array(
            [
                (
                    [0.8, 0.1, 0.2],
                    [0.9, 0.1, 0.2],
                    [0.3, 0.2, 0.8],
                    [0.1, 0.2, 0.8],
                ),
                (
                    [0.1, 0.8, 0.3],
                    [0.0, 0.8, 0.2],
                    [0.2, 0.1, 0.9],
                    [0.0, 0.2, 0.8],
                ),
                (
                    [0.1, 0.7, 0.3],
                    [0.3, 0.8, 0.2],
                    [0.1, 0.2, 0.8],
                    [0.4, 0.3, 0.7],
                ),
                (
                    [0.0, 1.0, 0.0],
                    [0.1, 0.9, 0.1],
                    [0.1, 0.1, 0.9],
                    [0.2, 0.2, 0.8],
                ),
            ]
        )
        hard_predictions = np.array([(0, 0, 2, 2), (1, 1, 2, 2), (1, 1, 2, 2), (1, 1, 2, 2)])

        metadata = {
            "non-required key": 1,
            "other non-required key": 2,
            "soft_predictions": soft_predictions,
        }

        predictions_to_annotations = converter.convert_to_annotation(predictions=hard_predictions, metadata=metadata)
        check_annotation_scene(annotation_scene=predictions_to_annotations, expected_length=2)
        check_annotation(
            actual_annotation=predictions_to_annotations.annotations[0],
            expected_label=labels[0],
            expected_probability=0.8333333333333333,
            expected_points=[
                Point(0.0, 0.25),
                Point(0.0, 0.5),
                Point(0.0, 0.75),
                Point(0.25, 0.75),
                Point(0.25, 0.5),
                Point(0.25, 0.25),
            ],
        )
        check_annotation(
            actual_annotation=predictions_to_annotations.annotations[1],
            expected_label=labels[1],
            expected_probability=0.8125,
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
        )

    @pytest.mark.components(OtxSdkComponent.OTX_API)
    class TestClassificationToAnnotation:
        @pytest.mark.priority_medium
        @pytest.mark.unit
        @pytest.mark.reqids(Requirements.REQ_1)
        def test_classification_to_annotation_init(self):
            """
            <b>Description:</b>
            Check "ClassificationToAnnotationConverter" class object initialization

            <b>Input data:</b>
            "ClassificationToAnnotationConverter" class object with specified "label_schema" parameter

            <b>Expected results:</b>
            Test passes if attributes of initialized "ClassificationToAnnotationConverter" object are equal to
            expected

            <b>Steps</b>
            1. Check attributes of "ClassificationToAnnotationConverter" object initialized with one label group
            with non-empty labels list length more than 1
            2. Check attributes of "ClassificationToAnnotationConverter" object initialized with one label group
            with non-empty labels list length equal to 1
            3. Check attributes of "ClassificationToAnnotationConverter" object initialized with two label groups
            with one label in each
            4. Check attributes of "ClassificationToAnnotationConverter" object initialized with two label groups
            with several labels in each
            """
            label_0 = LabelEntity(name="label_0", domain=Domain.CLASSIFICATION, id=ID("0"))
            label_0_1 = LabelEntity(name="label_0_1", domain=Domain.CLASSIFICATION, id=ID("0_1"))
            label_0_2 = LabelEntity(name="label_0_2", domain=Domain.CLASSIFICATION, id=ID("0_2"))
            label_0_1_1 = LabelEntity(name="label_0_1_1", domain=Domain.CLASSIFICATION, id=ID("0_1_1"))

            non_empty_labels = [label_0, label_0_1, label_0_1_1, label_0_2]
            empty_labels = [
                LabelEntity(
                    name="empty label",
                    domain=Domain.CLASSIFICATION,
                    is_empty=True,
                    id=ID("3"),
                )
            ]
            # Checking attributes of "ClassificationToAnnotationConverter" initialized with one label group with
            # non-empty labels list length more than 1
            label_group = LabelGroup(
                name="Classification labels group",
                labels=non_empty_labels + empty_labels,
            )
            label_schema = LabelSchemaEntity(label_groups=[label_group])
            converter = ClassificationToAnnotationConverter(label_schema=label_schema)
            assert converter.labels == non_empty_labels
            assert converter.empty_label == empty_labels[0]
            assert converter.label_schema == label_schema
            assert not converter.hierarchical
            # Checking attributes of "ClassificationToAnnotationConverter" initialized with one label group with
            # non-empty labels list length equal to 1
            label_group = LabelGroup(name="Classification labels group", labels=[label_0] + empty_labels)
            label_schema = LabelSchemaEntity(label_groups=[label_group])
            converter = ClassificationToAnnotationConverter(label_schema=label_schema)
            assert converter.labels == [label_0] + empty_labels
            assert converter.empty_label == empty_labels[0]
            assert converter.label_schema == label_schema
            assert not converter.hierarchical
            # Checking attributes of "ClassificationToAnnotationConverter" initialized with two label groups with
            # one label in each
            label_group = LabelGroup(name="Classification labels group", labels=[label_0_1])
            other_label_group = LabelGroup(name="Other Classification labels group", labels=[label_0_2])
            label_schema = LabelSchemaEntity(label_groups=[label_group, other_label_group])
            converter = ClassificationToAnnotationConverter(label_schema=label_schema)
            assert converter.labels == [label_0_1, label_0_2]
            assert not converter.empty_label
            assert converter.label_schema == label_schema
            assert not converter.hierarchical
            # Checking attributes of "ClassificationToAnnotationConverter" initialized with two label groups with
            # several labels in each
            other_non_empty_labels = [
                LabelEntity(name="label", domain=Domain.CLASSIFICATION, id=ID("3")),
                LabelEntity(name="other label", domain=Domain.CLASSIFICATION, id=ID("4")),
            ]
            label_group = LabelGroup(name="Classification labels group", labels=non_empty_labels)
            other_label_group = LabelGroup(
                name="Other Classification labels group",
                labels=other_non_empty_labels,
            )
            label_schema = LabelSchemaEntity(label_groups=[label_group, other_label_group])
            converter = ClassificationToAnnotationConverter(label_schema=label_schema)
            assert converter.labels == non_empty_labels + other_non_empty_labels
            assert not converter.empty_label
            assert converter.label_schema == label_schema
            assert converter.hierarchical

        @pytest.mark.priority_medium
        @pytest.mark.unit
        @pytest.mark.reqids(Requirements.REQ_1)
        def test_classification_to_annotation_convert(self):
            """
            <b>Description:</b>
            Check "ClassificationToAnnotationConverter" class "convert_to_annotation" method

            <b>Input data:</b>
            "ClassificationToAnnotationConverter" class object, "predictions" list, "metadata" dictionary

            <b>Expected results:</b>
            Test passes if "AnnotationSceneEntity" object returned by "convert_to_annotation" method is equal to
            expected

            <b>Steps</b>
            1. Check attributes of "AnnotationSceneEntity" object returned by "convert_to_annotation" method for
            "ClassificationToAnnotationConverter" object initialized with label group with several non-empty labels
            2. Check attributes of "AnnotationSceneEntity" object returned by "convert_to_annotation" method with
            "predictions" parameter equal to empty list
            3. Check attributes of "AnnotationSceneEntity" object returned by "convert_to_annotation" method for
            "ClassificationToAnnotationConverter" object initialized with several LabelGroups
            """

            def check_annotation(actual_annotation: Annotation, expected_labels: list):
                assert isinstance(actual_annotation, Annotation)
                assert actual_annotation.get_labels(include_empty=True) == expected_labels
                assert isinstance(actual_annotation.shape, Rectangle)
                assert Rectangle.is_full_box(rectangle=actual_annotation.shape)

            label_0 = LabelEntity(name="label_0", domain=Domain.CLASSIFICATION, id=ID("0"))
            label_0_1 = LabelEntity(name="label_0_1", domain=Domain.CLASSIFICATION, id=ID("0_1"))
            label_0_2 = LabelEntity(name="label_0_2", domain=Domain.CLASSIFICATION, id=ID("0_2"))
            label_0_1_1 = LabelEntity(name="label_0_1_1", domain=Domain.CLASSIFICATION, id=ID("0_1_1"))
            non_empty_labels = [label_0, label_0_1, label_0_1_1, label_0_2]
            empty_labels = [
                LabelEntity(
                    name="empty label",
                    domain=Domain.CLASSIFICATION,
                    is_empty=True,
                    id=ID("3"),
                )
            ]
            # Checking "AnnotationSceneEntity" returned by "convert_to_annotation" for
            # "ClassificationToAnnotationConverter" initialized with label group with several non-empty labels
            label_group = LabelGroup(
                name="Classification labels group",
                labels=non_empty_labels + empty_labels,
            )
            label_schema = LabelSchemaEntity(label_groups=[label_group])
            label_schema.add_child(parent=label_0, child=label_0_1)
            label_schema.add_child(parent=label_0, child=label_0_2)
            label_schema.add_child(parent=label_0_1, child=label_0_1_1)
            converter = ClassificationToAnnotationConverter(label_schema=label_schema)
            predictions = [(0, 0.9), (1, 0.8), (2, 0.94), (3, 0.86)]
            predictions_to_annotations = converter.convert_to_annotation(predictions)
            check_annotation_scene(annotation_scene=predictions_to_annotations, expected_length=1)
            check_annotation(
                actual_annotation=predictions_to_annotations.annotations[0],
                expected_labels=[
                    ScoredLabel(label=label_0, probability=0.9),
                    ScoredLabel(label=label_0_1, probability=0.8),
                    ScoredLabel(label=label_0_1_1, probability=0.94),
                    ScoredLabel(label=label_0_2, probability=0.86),
                ],
            )
            # Checking attributes of "AnnotationSceneEntity" returned by "convert_to_annotation" method with
            # "predictions" equal to empty list
            converter = ClassificationToAnnotationConverter(label_schema=label_schema)
            predictions = []
            predictions_to_annotations = converter.convert_to_annotation(predictions)
            check_annotation_scene(annotation_scene=predictions_to_annotations, expected_length=1)
            check_annotation(
                actual_annotation=predictions_to_annotations.annotations[0],
                expected_labels=[ScoredLabel(label=empty_labels[0], probability=1.0)],
            )
            # Checking attributes of "AnnotationSceneEntity" returned by "convert_to_annotation" for
            # "ClassificationToAnnotationConverter" initialized with several LabelGroups
            label_group = LabelGroup(name="Classification labels group", labels=[label_0_1_1])
            other_label_group = LabelGroup(
                name="Other Classification labels group",
                labels=[label_0_1, label_0_2],
            )
            label_schema = LabelSchemaEntity(label_groups=[label_group, other_label_group])

            label_schema.add_child(parent=label_0_1, child=label_0_1_1)
            converter = ClassificationToAnnotationConverter(label_schema=label_schema)
            predictions = [(2, 0.9), (1, 0.8)]
            predictions_to_annotations = converter.convert_to_annotation(predictions)
            check_annotation_scene(annotation_scene=predictions_to_annotations, expected_length=1)
            check_annotation(
                predictions_to_annotations.annotations[0],
                expected_labels=[ScoredLabel(label=label_0_2, probability=0.9)],
            )

    @pytest.mark.components(OtxSdkComponent.OTX_API)
    class TestAnomalyClassificationToAnnotation:
        @pytest.mark.priority_medium
        @pytest.mark.unit
        @pytest.mark.reqids(Requirements.REQ_1)
        def test_anomaly_classification_to_annotation_init(
            self,
        ):
            """
            <b>Description:</b>
            Check "AnomalyClassificationToAnnotationConverter" class initialization

            <b>Input data:</b>
            "AnomalyClassificationToAnnotationConverter" class object with specified "label_schema" parameter

            <b>Expected results:</b>
            Test passes if attributes of initialized "AnomalyClassificationToAnnotationConverter" object are equal
            to expected

            <b>Steps</b>
            1. Check attributes of "AnomalyClassificationToAnnotationConverter" object initialized with non-empty
            labels list
            2. Check attributes of "AnomalyClassificationToAnnotationConverter" object initialized with non-empty
            and empty labels list
            """
            # Checking attributes of "AnomalyClassificationToAnnotationConverter" initialized with non-empty labels
            # list
            non_empty_labels = [
                LabelEntity(name="Normal", domain=Domain.CLASSIFICATION, id=ID("1")),
                LabelEntity(name="Normal", domain=Domain.CLASSIFICATION, id=ID("2")),
                LabelEntity(
                    name="Anomalous",
                    domain=Domain.CLASSIFICATION,
                    id=ID("1"),
                    is_anomalous=True,
                ),
                LabelEntity(
                    name="Anomalous",
                    domain=Domain.CLASSIFICATION,
                    id=ID("2"),
                    is_anomalous=True,
                ),
            ]
            label_group = LabelGroup(name="Classification labels group", labels=non_empty_labels)
            label_schema = LabelSchemaEntity(label_groups=[label_group])
            converter = AnomalyClassificationToAnnotationConverter(label_schema=label_schema)
            assert converter.normal_label == non_empty_labels[0]
            assert converter.anomalous_label == non_empty_labels[2]
            # Checking attributes of "AnomalyClassificationToAnnotationConverter" initialized with non-empty and
            # empty labels list
            empty_labels = [
                LabelEntity(
                    name="Normal",
                    domain=Domain.CLASSIFICATION,
                    is_empty=True,
                    id=ID("3"),
                ),
                LabelEntity(
                    name="Normal",
                    domain=Domain.CLASSIFICATION,
                    is_empty=True,
                    id=ID("4"),
                ),
                LabelEntity(
                    name="Anomalous",
                    domain=Domain.CLASSIFICATION,
                    is_empty=True,
                    id=ID("3"),
                ),
                LabelEntity(
                    name="Anomalous",
                    domain=Domain.CLASSIFICATION,
                    is_empty=True,
                    id=ID("4"),
                ),
            ]
            label_group = LabelGroup(
                name="Anomaly classification labels group",
                labels=non_empty_labels + empty_labels,
            )
            label_schema = LabelSchemaEntity(label_groups=[label_group])
            converter = AnomalyClassificationToAnnotationConverter(label_schema=label_schema)
            assert converter.normal_label == non_empty_labels[0]
            assert converter.anomalous_label == non_empty_labels[2]

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_anomaly_classification_to_annotation_convert(
        self,
    ):
        """
        <b>Description:</b>
        Check "AnomalyClassificationToAnnotationConverter" class "convert_to_annotation" method

        <b>Input data:</b>
        "AnomalyClassificationToAnnotationConverter" class object, "predictions" array

        <b>Expected results:</b>
        Test passes if "AnnotationSceneEntity" object returned by "convert_to_annotation" method is equal to
        expected

        <b>Steps</b>
        1. Check attributes of "AnnotationSceneEntity" object returned by "convert_to_annotation" method for
        "metadata" dictionary with specified "threshold" key
        2. Check attributes of "AnnotationSceneEntity" object returned by "convert_to_annotation" method for
        "metadata" dictionary without specified "threshold" key
        """

        def check_annotation(actual_annotation: Annotation, expected_labels: list):
            assert isinstance(actual_annotation, Annotation)
            assert actual_annotation.get_labels() == expected_labels
            assert isinstance(actual_annotation.shape, Rectangle)
            assert Rectangle.is_full_box(rectangle=actual_annotation.shape)

        non_empty_labels = [
            LabelEntity(name="Normal", domain=Domain.CLASSIFICATION, id=ID("1")),
            LabelEntity(
                name="Anomalous",
                domain=Domain.CLASSIFICATION,
                id=ID("2"),
                is_anomalous=True,
            ),
        ]
        label_group = LabelGroup(name="Anomaly classification labels group", labels=non_empty_labels)
        label_schema = LabelSchemaEntity(label_groups=[label_group])
        converter = AnomalyClassificationToAnnotationConverter(label_schema=label_schema)
        predictions = np.array([0.7])
        # Checking attributes of "AnnotationSceneEntity" returned by "convert_to_annotation" for "metadata" with
        # specified "threshold" key
        metadata = {
            "non-required key": 1,
            "other non-required key": 2,
            "threshold": 0.8,
        }
        predictions_to_annotations = converter.convert_to_annotation(predictions=predictions, metadata=metadata)
        check_annotation_scene(annotation_scene=predictions_to_annotations, expected_length=1)
        check_annotation(
            predictions_to_annotations.annotations[0],
            expected_labels=[ScoredLabel(label=non_empty_labels[0], probability=0.7)],
        )
        # Checking attributes of "AnnotationSceneEntity" returned by "convert_to_annotation" for "metadata" without
        # specified "threshold" key
        metadata = {"non-required key": 1, "other non-required key": 2}
        predictions_to_annotations = converter.convert_to_annotation(predictions=predictions, metadata=metadata)
        check_annotation_scene(annotation_scene=predictions_to_annotations, expected_length=1)
        check_annotation(
            predictions_to_annotations.annotations[0],
            expected_labels=[ScoredLabel(label=non_empty_labels[1], probability=0.7)],
        )
