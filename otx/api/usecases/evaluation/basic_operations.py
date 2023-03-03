"""This module contains functions for basic operations."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


from typing import Dict, List, Optional, Tuple

import numpy as np

from otx.api.entities.label import LabelEntity
from otx.api.entities.shapes.rectangle import Rectangle

#: Dictionary storing a number for each label. The ``None`` key represents "all labels"
NumberPerLabel = Dict[Optional[LabelEntity], int]


def get_intersections_and_cardinalities(
    references: List[np.ndarray],
    predictions: List[np.ndarray],
    labels: List[LabelEntity],
) -> Tuple[NumberPerLabel, NumberPerLabel]:
    """Returns all intersections and cardinalities between reference masks and prediction masks.

    Intersections and cardinalities are each returned in a dictionary mapping each label to its corresponding
    number of intersection/cardinality pixels

    Args:
        references (List[np.ndarray]): reference masks,s one mask per image
        predictions (List[np.ndarray]): prediction masks, one mask per image
        labels (List[LabelEntity]): labels in input masks

    Returns:
        Tuple[NumberPerLabel, NumberPerLabel]: (all_intersections, all_cardinalities)
    """

    # TODO [Soobee] : Add score for background label and align the calculation method with validation
    all_intersections: NumberPerLabel = {label: 0 for label in labels}
    all_intersections[None] = 0
    all_cardinalities: NumberPerLabel = {label: 0 for label in labels}
    all_cardinalities[None] = 0
    for reference, prediction in zip(references, predictions):
        intersection = np.where(reference == prediction, reference, 0)
        all_intersections[None] += np.count_nonzero(intersection)
        all_cardinalities[None] += np.count_nonzero(reference) + np.count_nonzero(prediction)
        for i, label in enumerate(labels):
            label_num = i + 1
            all_intersections[label] += np.count_nonzero(intersection == label_num)
            reference_area = np.count_nonzero(reference == label_num)
            prediction_area = np.count_nonzero(prediction == label_num)
            all_cardinalities[label] += reference_area + prediction_area
    return all_intersections, all_cardinalities


def intersection_box(box1: Rectangle, box2: Rectangle) -> Optional[List[float]]:
    """Calculate the intersection box of two bounding boxes.

    Args:
        box1: a Rectangle that represents the first bounding box
        box2: a Rectangle that represents the second bounding box

    Returns:
        a Rectangle that represents the intersection box if inputs have
        a valid intersection, else None
    """
    x_left = max(box1.x1, box2.x1)
    y_top = max(box1.y1, box2.y1)
    x_right = min(box1.x2, box2.x2)
    y_bottom = min(box1.y2, box2.y2)
    if x_right <= x_left or y_bottom <= y_top:
        return None
    return [x_left, y_top, x_right, y_bottom]


def intersection_over_union(box1: Rectangle, box2: Rectangle, intersection: Optional[List[float]] = None) -> float:
    """Calculate the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1: a Rectangle representing a bounding box
        box2: a Rectangle representing a second bounding box
        intersection: precomputed intersection between two boxes (see
            intersection_box function), if exists.

    Returns:
        intersection-over-union of box1 and box2
    """
    iou = 0.0
    if intersection is None:
        intersection = intersection_box(box1, box2)
    if intersection is not None:
        intersection_area = (intersection[2] - intersection[0]) * (intersection[3] - intersection[1])
        box1_area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        box2_area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
        union_area = float(box1_area + box2_area - intersection_area)
        if union_area != 0:
            iou = intersection_area / union_area
    if iou < 0.0 or iou > 1.0:
        raise ValueError(f"intersection over union should be in range [0,1], instead got iou={iou}")
    return iou


def precision_per_class(matrix: np.ndarray) -> np.ndarray:
    """Compute the precision per class based on the confusion matrix.

    Args:
        matrix: the computed confusion matrix

    Returns:
        the precision (per class), defined as TP/(TP+FP)
    """
    if not matrix.shape[0] == matrix.shape[1]:
        # If the matrix is not square (there is a column for "other" label), the "other" column is deleted.
        # Otherwise, there will be 3 elements in TP and 4 in TP+FP meaning they can't be divided.
        matrix = np.delete(matrix, -1, 1)

    tp_per_class = matrix.diagonal()
    sum_tp_fp_per_class = matrix.sum(0)
    return divide_arrays_with_possible_zeros(tp_per_class, sum_tp_fp_per_class)


def recall_per_class(matrix: np.ndarray) -> np.ndarray:
    """Compute the recall per class based on the confusion matrix.

    Args:
        matrix: the computed confusion matrix

    Returns:
        the recall (per class), defined as TP/(TP+FN)
    """
    tp_per_class = matrix.diagonal()
    sum_tp_fn_per_class = matrix.sum(1)
    return divide_arrays_with_possible_zeros(tp_per_class, sum_tp_fn_per_class)


def divide_arrays_with_possible_zeros(array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
    """Sometimes the denominator in the precision or recall computation can contain a zero.

    In that case, a zero is returned for that element (https://stackoverflow.com/a/32106804).

    Args:
        array1: the numerator
        array2: the denominator

    Returns:
        the divided arrays (numerator/denominator) with a value of zero
        where the denominator was zero.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.true_divide(array1, array2)
        result[result == np.inf] = 0  # If the denominator is a float, np.inf is returned
        result = np.nan_to_num(result)  # If the denominator is an int, np.nan is returned
    return result
