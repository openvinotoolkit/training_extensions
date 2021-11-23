"""
This module implements segmentation related utilities
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

import warnings
from typing import List

import cv2
import numpy as np
from bson import ObjectId

from ote_sdk.entities.annotation import Annotation
from ote_sdk.entities.dataset_item import DatasetItemEntity
from ote_sdk.entities.id import ID
from ote_sdk.entities.label import LabelEntity
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.shapes.polygon import Point, Polygon
from ote_sdk.utils.shape_factory import ShapeFactory
from ote_sdk.utils.time_utils import timeit


def mask_from_dataset_item(
    dataset_item: DatasetItemEntity, labels: List[LabelEntity]
) -> np.ndarray:
    """
    Creates a mask from dataset item. The mask will be two dimensional,
    and the value of each pixel matches the class index with offset 1. The background
    class index is zero. labels[0] matches pixel value 1, etc. The class index is
    determined based on the order of :param labels:

    :param dataset_item: Item to make mask for
    :param labels: The labels to use for creating the mask. The order of the labels
                   determines the class index.

    :return: Numpy array of mask
    """
    # todo: cache this so that it does not have to be redone for all the same media
    mask = mask_from_annotation(
        dataset_item.get_annotations(),
        labels,
        dataset_item.width,
        dataset_item.height,
    )

    return mask


def mask_from_annotation(
    annotations: List[Annotation],
    labels: List[LabelEntity],
    width: int,
    height: int,
) -> np.ndarray:
    """
    Generate a segmentation mask of a numpy image, and a list of shapes.
    The mask is will be two dimensional    and the value of each pixel matches the class
    index with offset 1. The background class index is zero.  labels[0] matches pixel
    value 1, etc. The class index is determined based on the order of :param labels:

    :param annotations: List of annotations to plot in mask
    :param labels: List of labels. The index position of the label determines the class
                   number in the segmentation mask.
    :param width: Width of the mask
    :param height: Height of the mask

    :return: 2d numpy array of mask
    """

    labels = sorted(labels)  # type: ignore
    mask = np.zeros(shape=(height, width), dtype=np.uint8)
    for annotation in annotations:
        shape = annotation.shape
        if not isinstance(shape, Polygon):
            shape = ShapeFactory.shape_as_polygon(annotation.shape)
        known_labels = [
            label
            for label in annotation.get_labels()
            if isinstance(label, ScoredLabel) and label.get_label() in labels
        ]
        if len(known_labels) == 0:
            # Skip unknown shapes
            continue

        label_to_compare = known_labels[0].get_label()

        class_idx = labels.index(label_to_compare) + 1
        contour = []
        for point in shape.points:
            contour.append([int(point.x * width), int(point.y * height)])

        mask = cv2.drawContours(
            mask,
            np.asarray([contour]),
            0,
            (class_idx, class_idx, class_idx),
            -1,
        )

    mask = np.expand_dims(mask, axis=2)

    return mask


def create_hard_prediction_from_soft_prediction(
    soft_prediction: np.ndarray, soft_threshold: float, blur_strength: int = 5
) -> np.ndarray:
    """
    Creates a hard prediction containing the final label index per pixel

    :param soft_prediction: Output from segmentation network. Assumes floating point
                            values, between 0.0 and 1.0. Can be a 2d-array of shape
                            (height, width) or per-class segmentation logits of shape
                            (height, width, num_classes)
    :param soft_threshold: minimum class confidence for each pixel.
                            The higher the value, the more strict the segmentation is
                            (usually set to 0.5)
    :param blur_strength: The higher the value, the smoother the segmentation output
                            will be, but less accurate
    :return: Numpy array of the hard prediction
    """
    soft_prediction_blurred = cv2.blur(soft_prediction, (blur_strength, blur_strength))
    if len(soft_prediction.shape) == 3:
        # Apply threshold to filter out `unconfident` predictions, then get max along
        # class dimension
        soft_prediction_blurred[soft_prediction_blurred < soft_threshold] = 0
        hard_prediction = np.argmax(soft_prediction_blurred, axis=2)
    elif len(soft_prediction.shape) == 2:
        # In the binary case, simply apply threshold
        hard_prediction = soft_prediction_blurred > soft_threshold
    else:
        raise ValueError(
            f"Invalid prediction input of shape {soft_prediction.shape}. "
            f"Expected either a 2D or 3D array."
        )
    return hard_prediction


@timeit
def create_annotation_from_segmentation_map(
    hard_prediction: np.ndarray, soft_prediction: np.ndarray, label_map: dict
) -> List[Annotation]:
    """
    Creates polygons from the soft predictions.
    Background label will be ignored and not be converted to polygons.

    :param hard_prediction: hard prediction containing the final label index per pixel.
        See function `create_hard_prediction_from_soft_prediction`.
    :param soft_prediction: soft prediction with shape H x W x N_labels,
        where soft_prediction[:, :, 0] is the soft prediction for background.
        If soft_prediction is of H x W shape, it is assumed that this soft prediction
        will be applied for all labels.
    :param label_map: dictionary mapping labels to an index.
        It is assumed that the first item in the dictionary corresponds to the
        background label and will therefore be ignored.
    :return: List of shapes
    """
    # pylint: disable=too-many-locals
    height, width = hard_prediction.shape[:2]
    img_class = hard_prediction.swapaxes(0, 1)

    annotations: List[Annotation] = []
    for label_index, label in label_map.items():
        # Skip background
        if label_index == 0:
            continue

        # obtain current label soft prediction
        if len(soft_prediction.shape) == 3:
            current_label_soft_prediction = soft_prediction[:, :, label_index]
        else:
            current_label_soft_prediction = soft_prediction

        obj_group = img_class == label_index
        label_index_map = (obj_group.T.astype(int) * 255).astype(np.uint8)

        # Contour retrieval mode CCOMP (Connected components) creates a two-level
        # hierarchy of contours
        contours, hierarchies = cv2.findContours(
            label_index_map, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )

        if hierarchies is not None:
            for contour, hierarchy in zip(contours, hierarchies[0]):
                if hierarchy[3] == -1:
                    # In this case a contour does not represent a hole
                    points = [
                        Point(x=point[0][0] / width, y=point[0][1] / height)
                        for point in contour
                    ]

                    # compute probability of the shape
                    mask = np.zeros(hard_prediction.shape, dtype=np.uint8)
                    cv2.drawContours(
                        mask, contour, contourIdx=-1, color=1, thickness=-1
                    )
                    probability = cv2.mean(current_label_soft_prediction, mask)[0]

                    if len(list(contour)) > 2:
                        # Contour is a closed polygon
                        annotations.append(
                            Annotation(
                                Polygon(points=points),
                                labels=[ScoredLabel(label, probability)],
                                id=ID(ObjectId()),
                            )
                        )
                    else:
                        # Contour is a single point or a free-standing line
                        # Give a warning if one dimensional elements will be deleted
                        # from the annotation
                        warnings.warn(
                            "The geometry of the segmentation map you are converting "
                            "is not fully supported. Points or lines with a linewidth "
                            "of 1 pixel will be removed.",
                            UserWarning,
                        )

                else:
                    # If contour hierarchy[3] != -1 then contour has a parent and
                    # therefore is a hole
                    # Do not allow holes in segmentation masks to be filled silently,
                    # but trigger warning instead
                    warnings.warn(
                        "The geometry of the segmentation map you are converting is "
                        "not fully supported. A hole was found and will be filled.",
                        UserWarning,
                    )

    return annotations
