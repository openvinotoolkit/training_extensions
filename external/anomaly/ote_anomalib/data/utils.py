"""OTE Dataset Utilities."""

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

import cv2
import numpy as np

from ote_sdk.entities.annotation import (
    Annotation,
)
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.shapes.polygon import Point, Polygon
from ote_sdk.entities.shapes.rectangle import Rectangle


def annotations_from_mask(mask: np.ndarray, normal_label, anomalous_label):
    # TODO: add anomaly_map argument to extract confidence scores
    height, width = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    annotations = []

    for contour in contours:
        points = list((point[0][0] / width, point[0][1] / height) for point in contour)
        points = [Point(x=x, y=y) for x, y in points]

        polygon = Polygon(points=points)
        annotations.append(
            Annotation(
                shape=polygon,
                labels=[ScoredLabel(anomalous_label, 1.0)],
            )
        )
    if len(annotations) == 0:
        # TODO: add confidence to this label
        annotations = [
            Annotation(
                Rectangle.generate_full_box(),
                labels=[ScoredLabel(label=normal_label, probability=1.0)],
            )
        ]

    return annotations
