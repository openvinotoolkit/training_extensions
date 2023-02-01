"""Detection Utils."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List

import cv2
import numpy as np

from otx.api.entities.annotation import Annotation
from otx.api.entities.label import LabelEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.rectangle import Rectangle


def create_detection_annotation_from_anomaly_heatmap(
    hard_prediction: np.ndarray,
    soft_prediction: np.ndarray,
    label_map: Dict[int, LabelEntity],
) -> List[Annotation]:
    """Create box annotation from the soft predictions.

    Args:
        hard_prediction: hard prediction containing the final label
            index per pixel.
        soft_prediction: soft prediction with shape
        label_map: dictionary mapping labels to an index. It is assumed
            that the first item in the dictionary corresponds to the
            background label and will therefore be ignored.

    Returns:
        List of annotations.
    """
    # pylint: disable=too-many-locals
    if hard_prediction.ndim == 3 and hard_prediction.shape[0] == 1:
        hard_prediction = hard_prediction.squeeze().astype(np.uint8)

    if soft_prediction.ndim == 3 and soft_prediction.shape[0] == 1:
        soft_prediction = soft_prediction.squeeze()

    image_h, image_w = hard_prediction.shape[:2]

    annotations: List[Annotation] = []
    for label_index, label in label_map.items():
        # Skip the normal label.
        if label_index == 0:
            continue

        # cv2.connectedComponentsWithStats returns num_labels, labels, coordinates
        # and centroids. This script only needs the coordinates.
        _, connected_components, coordinates, _ = cv2.connectedComponentsWithStats(hard_prediction)

        for i, coordinate in enumerate(coordinates):
            # First row of the coordinates is always backround,
            # so should be ignored.
            if i == 0:
                continue

            # Last column of the coordinates is the area of the connected component.
            # It could therefore be ignored.
            comp_x, comp_y, comp_w, comp_h, _ = coordinate

            # Compute the probability of each connected-component
            component_hard_prediction = (connected_components == i).astype(np.uint8)
            component_soft_prediction = cv2.bitwise_and(
                soft_prediction, soft_prediction, mask=component_hard_prediction
            )

            # NOTE: Find the best approach to calculate the probability
            probability = component_soft_prediction.reshape(-1).max()

            # NOTE: NMS could be needed here.

            # Create the annotation based on the box shape and the probability.
            shape = Rectangle(
                x1=comp_x / image_w,
                y1=comp_y / image_h,
                x2=(comp_x + comp_w) / image_w,
                y2=(comp_y + comp_h) / image_h,
            )
            annotations.append(Annotation(shape=shape, labels=[ScoredLabel(label, float(probability))]))
    return annotations
