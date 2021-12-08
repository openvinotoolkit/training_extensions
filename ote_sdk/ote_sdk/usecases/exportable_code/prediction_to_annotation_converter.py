"""
Converters for output of inferencers
"""

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

import abc
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from openvino.model_zoo.model_api.models import utils

from ote_sdk.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from ote_sdk.entities.id import ID
from ote_sdk.entities.label import Domain, LabelEntity
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.utils.segmentation_utils import create_annotation_from_segmentation_map
from ote_sdk.utils.time_utils import now


class IPredictionToAnnotationConverter(metaclass=abc.ABCMeta):
    """
    Interface for converter
    """

    @abc.abstractmethod
    def convert_to_annotation(
        self, predictions: Any, metadata: Dict
    ) -> AnnotationSceneEntity:
        """
        Convert raw predictions to AnnotationScene format.

        :param predictions: Raw predictions from the model
        :return: annotation object containing the shapes
                 obtained from the raw predictions.
        """
        raise NotImplementedError


class DetectionToAnnotationConverter(IPredictionToAnnotationConverter):
    """
    Converts Object Detections to Annotations
    """

    def __init__(self, labels: List[LabelEntity]):
        self.label_map = dict(enumerate(labels))

    def convert_to_annotation(
        self, predictions: np.ndarray, metadata: Optional[Dict] = None
    ) -> AnnotationSceneEntity:
        """
        Converts a set of predictions into an AnnotationScene object

        :param predictions: Prediction with shape [num_predictions, 6] or
                            [num_predictions, 7]
        Supported detection formats are

        * [label, confidence, x1, y1, x2, y2]
        * [_, label, confidence, x1, y1, x2, y2]

        .. note::
           `label` can be any integer that can be mapped to `self.labels`
           `confidence` should be a value between 0 and 1
           `x1`, `x2`, `y1` and `y2` are expected to be normalized.

        :returns AnnotationScene: AnnotationScene Object containing the boxes
                                  obtained from the prediction
        """
        annotations = self.__convert_to_annotations(predictions)
        # media_identifier = ImageIdentifier(image_id=ID())
        annotation_scene = AnnotationSceneEntity(
            id=ID(),
            kind=AnnotationSceneKind.PREDICTION,
            editor="ote",
            creation_date=now(),
            annotations=annotations,
        )

        return annotation_scene

    def __convert_to_annotations(self, predictions: np.ndarray) -> List[Annotation]:
        """
        Converts a list of Detections to OTE SDK Annotation objects

        :param predictions: A list of predictions with shape [num_prediction, 6] or
                            [num_predictions, 7]

        :returns: A list of Annotation objects with Rectangle shapes

        :raises ValueError: This error is raised if the shape of prediction is not
                            (n, 7) or (n, 6)
        """
        annotations = []
        if predictions.shape[1:] < (6,) or predictions.shape[1:] > (7,):
            raise ValueError(
                f"Shape of prediction is not expected, expected (n, 7) or (n, 6) "
                f"got {predictions.shape}"
            )

        for prediction in predictions:

            if prediction.shape == (7,):
                # Some OpenVINO models use an output shape of [7,]
                # If this is the case, skip the first value as it is not used
                prediction = prediction[1:]

            label = int(prediction[0])
            confidence = prediction[1]
            scored_label = ScoredLabel(self.label_map[label], confidence)
            annotations.append(
                Annotation(
                    Rectangle(
                        prediction[2], prediction[3], prediction[4], prediction[5]
                    ),
                    labels=[scored_label],
                )
            )

        return annotations


def create_converter(
    converter_type: Domain, labels: List[Union[str, LabelEntity]]
) -> IPredictionToAnnotationConverter:
    """
    Simple factory for converters based on type of tasks
    """

    converter: IPredictionToAnnotationConverter
    if converter_type == Domain.DETECTION:
        converter = DetectionBoxToAnnotationConverter(labels)
    elif converter_type == Domain.SEGMENTATION:
        converter = SegmentationToAnnotationConverter(labels)
    elif converter_type == Domain.CLASSIFICATION:
        converter = ClassificationToAnnotationConverter(labels)
    elif converter_type == Domain.ANOMALY_CLASSIFICATION:
        converter = AnomalyClassificationToAnnotationConverter(labels)
    else:
        raise ValueError(f"Unknown converter type: {converter_type}")

    return converter


def get_label(labels_map: List[Any], index: int, label_domain: Domain) -> LabelEntity:
    """
    Get label from list of labels
    """
    label = labels_map[index]
    if isinstance(label, LabelEntity):
        return label

    return LabelEntity(str(label), label_domain)


class DetectionBoxToAnnotationConverter(IPredictionToAnnotationConverter):
    """
    Converts DetectionBox Predictions ModelAPI to Annotations
    """

    def __init__(self, labels: List[Union[str, LabelEntity]]):
        self.labels_map = labels

    def convert_to_annotation(
        self, predictions: List[utils.Detection], metadata: Dict[str, Any]
    ) -> AnnotationSceneEntity:
        annotations = []
        image_size = metadata["original_shape"][1::-1]
        for box in predictions:
            scored_label = ScoredLabel(
                get_label(self.labels_map, int(box.id), Domain.DETECTION), box.score
            )
            coords = np.array(box.get_coords()) / np.tile(image_size, 2)
            annotations.append(
                Annotation(
                    Rectangle(coords[0], coords[1], coords[2], coords[3]),
                    labels=[scored_label],
                )
            )

        annotation_scene = AnnotationSceneEntity(
            kind=AnnotationSceneKind.PREDICTION,
            annotations=annotations,
        )
        return annotation_scene


class SegmentationToAnnotationConverter(IPredictionToAnnotationConverter):
    """
    Converts Segmentation Predictions ModelAPI to Annotations
    """

    def __init__(self, labels: List[Union[str, LabelEntity]]):
        if labels is None:
            raise ValueError("Labels for segmentation model is None")
        self.label_map = {
            i + 1: get_label(labels, i, Domain.SEGMENTATION) for i in range(len(labels))
        }

    def convert_to_annotation(
        self, predictions: np.ndarray, metadata: Dict[str, Any]
    ) -> AnnotationSceneEntity:
        soft_predictions = metadata.get("soft_predictions", np.ones(predictions.shape))
        annotations = create_annotation_from_segmentation_map(
            hard_prediction=predictions,
            soft_prediction=soft_predictions,
            label_map=self.label_map,
        )

        return AnnotationSceneEntity(
            kind=AnnotationSceneKind.PREDICTION, annotations=annotations
        )


class ClassificationToAnnotationConverter(IPredictionToAnnotationConverter):
    """
    Converts Classification Predictions ModelAPI to Annotations
    """

    def __init__(self, labels: List[Union[str, LabelEntity]]):
        self.labels_map = labels

    def convert_to_annotation(
        self, predictions: List[Tuple[int, float]], metadata: Dict[str, Any]
    ) -> AnnotationSceneEntity:
        labels = []
        for index, score in predictions:
            labels.append(
                ScoredLabel(
                    get_label(self.labels_map, index, Domain.CLASSIFICATION), score
                )
            )

        if not labels and metadata.get("empty_label") is not None:
            labels = [ScoredLabel(metadata["empty_label"], probability=1.0)]

        annotations = [Annotation(Rectangle.generate_full_box(), labels=labels)]
        return AnnotationSceneEntity(
            kind=AnnotationSceneKind.PREDICTION, annotations=annotations
        )


class AnomalyClassificationToAnnotationConverter(IPredictionToAnnotationConverter):
    """
    Converts AnomalyClassification Predictions ModelAPI to Annotations
    """

    def __init__(self, labels: List[Union[str, LabelEntity]]):
        self.normal_label = get_label(labels, 0, Domain.ANOMALY_CLASSIFICATION)
        self.anomalous_label = get_label(labels, 1, Domain.ANOMALY_CLASSIFICATION)

    def convert_to_annotation(
        self, predictions: np.ndarray, metadata: Dict[str, Any]
    ) -> AnnotationSceneEntity:
        pred_score = predictions.reshape(-1).max()
        pred_label = pred_score >= metadata.get("threshold", 0.5)
        assigned_label = self.anomalous_label if pred_label else self.normal_label

        annotations = [
            Annotation(
                Rectangle.generate_full_box(),
                labels=[ScoredLabel(assigned_label, probability=pred_score)],
            )
        ]
        return AnnotationSceneEntity(
            kind=AnnotationSceneKind.PREDICTION, annotations=annotations
        )
