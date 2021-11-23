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
from typing import Any, Dict, List, Union

import numpy as np

from ote_sdk.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from openvino.model_zoo.model_api.models import utils
from ote_sdk.entities.id import ID
from ote_sdk.entities.label import Domain, LabelEntity
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.utils.segmentation_utils import create_annotation_from_segmentation_map
from ote_sdk.utils.time_utils import now


class IPredictionToAnnotationConverter(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def convert_to_annotation(self, predictions: Any) -> AnnotationSceneEntity:
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

    def convert_to_annotation(self, predictions: np.ndarray) -> AnnotationSceneEntity:
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
        annotations = list()
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


def create_converter(type: Domain, labels: List[Union[str, LabelEntity]]):
    if type == Domain.DETECTION:
        return DetectionBoxToAnnotationConverter(labels)
    elif type == Domain.SEGMENTATION:
        return SegmentationToAnnotationConverter(labels)
    else:
        raise ValueError(type)


def get_label(labels_map: List[Union[str, LabelEntity]], id: int, label_domain: Domain) -> LabelEntity:
    if labels_map is None:
        return LabelEntity(id, label_domain)
    if isinstance(labels_map[id], str):
        return LabelEntity(labels_map[id], label_domain)

    return labels_map[id]


class DetectionBoxToAnnotationConverter(IPredictionToAnnotationConverter):
    """
    Converts DetectionBox Predictions ModelAPI to Annotations
    """

    def __init__(self, labels: List[Union[str, LabelEntity]]):
        self.labels_map = labels

    def convert_to_annotation(self, detections: List[utils.Detection], metadata: Dict[str, Any]) -> AnnotationSceneEntity:
        annotations = []
        image_size = metadata['original_shape'][1::-1]
        for box in detections:
            scored_label = ScoredLabel(get_label(self.labels_map, int(box.id), Domain.DETECTION), box.score)
            coords = np.array(box.get_coords()) / np.tile(image_size, 2)
            annotations.append(
                Annotation(
                    Rectangle(
                        coords[0], coords[1], coords[2], coords[3]
                    ),
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
        self.label_map = {i + 1: get_label(labels, i, Domain.SEGMENTATION) for i in range(len(labels))}

    def convert_to_annotation(self, hard_predictions: np.ndarray, metadata: Dict[str, Any]) -> AnnotationSceneEntity:
        soft_predictions = metadata.get('soft_predictions', np.ones(hard_predictions.shape))
        annotations = create_annotation_from_segmentation_map(
            hard_prediction=hard_predictions,
            soft_prediction=soft_predictions,
            label_map=self.label_map
        )

        return AnnotationSceneEntity(
            kind=AnnotationSceneKind.PREDICTION,
            annotations=annotations
        )
