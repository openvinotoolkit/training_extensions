"""Detection Utils."""

# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from typing import Dict, List

import cv2
import numpy as np
from ote_sdk.entities.annotation import Annotation
from ote_sdk.entities.label import LabelEntity
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.shapes.rectangle import Rectangle


def create_detection_annotation_from_anomaly_heatmap(
    hard_prediction: np.ndarray, soft_prediction: np.ndarray, label_map: Dict[int, LabelEntity]
) -> List[Annotation]:
    """Create box annotation from the soft predictions.

    Args:
        hard_prediction (np.ndarray): hard prediction containing the final label index per pixel.
        soft_prediction (np.ndarray): soft prediction with shape ,
        label_map (Dict[int, LabelEntity]): dictionary mapping labels to an index.
            It is assumed that the first item in the dictionary corresponds to the
            background label and will therefore be ignored.

    Returns:
        List[Annotation]: List of annotations.
    """
    # pylint: disable=too-many-locals
    if hard_prediction.ndim == 3 and hard_prediction.shape[0] == 1:
        hard_prediction = hard_prediction.squeeze().astype(np.uint8)

    if soft_prediction.ndim == 3 and soft_prediction.shape[0] == 1:
        soft_prediction = soft_prediction.squeeze()

    height, width = hard_prediction.shape[:2]

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
            x1, y1, x2, y2, _ = coordinate

            # Compute the probability of each connected-component
            component_hard_prediction = (connected_components == i).astype(np.uint8)
            component_soft_prediction = cv2.bitwise_and(
                soft_prediction, soft_prediction, mask=component_hard_prediction
            )

            # TODO: Find the best approach to calculate the probability
            # probability = cv2.mean(current_label_soft_prediction, component_soft_prediction)[0]
            probability = component_soft_prediction.reshape(-1).max()

            # TODO: Add NMS here.

            # Create the annotation based on the box shape and the probability.
            shape = Rectangle(x1=x1 / width, y1=y1 / height, x2=x2 / width, y2=y2 / height)
            annotations.append(Annotation(shape=shape, labels=[ScoredLabel(label, probability)]))
    return annotations
