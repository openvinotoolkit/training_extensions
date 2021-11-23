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
from typing import Any, List

import numpy as np

from ote_sdk.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from ote_sdk.entities.id import ID
from ote_sdk.entities.label import LabelEntity
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.shapes.rectangle import Rectangle
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
                        prediction[2],
                        prediction[3],
                        prediction[4],
                        prediction[5],
                    ),
                    labels=[scored_label],
                )
            )

        return annotations
