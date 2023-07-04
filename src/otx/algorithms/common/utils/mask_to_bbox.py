"""Convert a mask to a border image."""

# Copyright (C) 2021-2022 Intel Corporation
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

# pylint: disable=invalid-name

from typing import List

import numpy as np
from skimage.measure import find_contours, label, regionprops


def mask_to_border(mask):
    """Make a border by using a binary mask.

    Args:
        mask (np.ndarray): Input binary mask

    Returns:
        np.ndarray: Border image.
    """
    h, w = mask.shape
    border = np.zeros((h, w))

    contours = find_contours(mask, 0.5)  # since the input range is [0, 1], the threshold is 0.5
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 1  # since the input is binary, the value is 1

    return border


def mask2bbox(mask) -> List[List[int]]:
    """Mask to bounding boxes.

    Args:
        mask (np.ndarray): Input binary mask

    Returns:
        List[List[int]]: Bounding box coordinates
    """
    bboxes: List[List[int]] = []

    mask = mask_to_border(mask)
    print(np.unique(mask))
    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]

        x2 = prop.bbox[3]
        y2 = prop.bbox[2]

        bboxes.append([x1, y1, x2, y2])

    return bboxes
