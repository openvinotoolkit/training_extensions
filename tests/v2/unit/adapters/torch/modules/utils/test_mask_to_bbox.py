# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import numpy as np
from otx.v2.adapters.torch.modules.utils.mask_to_bbox import (
    convert_polygon_to_mask,
    generate_bbox,
    generate_bbox_from_mask,
)
from otx.v2.api.entities.shapes.polygon import Point, Polygon


def test_convert_polygon_to_mask() -> None:
    """Test convert_polygon_to_mask."""

    polygon = Polygon(points=[Point(x=0.1, y=0.1), Point(x=0.2, y=0.2), Point(x=0.3, y=0.3)])
    width = 100
    height = 100

    mask = convert_polygon_to_mask(polygon, width, height)

    assert isinstance(mask, np.ndarray)
    assert mask.shape == (height, width)
    assert mask.sum() == 21


def test_generate_bbox() -> None:
    """Test generate_bbox."""

    x1, y1, x2, y2 = 10, 20, 30, 40
    width = 100
    height = 100

    bbox = generate_bbox(x1, y1, x2, y2, width, height)

    assert isinstance(bbox, list)
    assert len(bbox) == 4
    assert bbox[0] >= 0
    assert bbox[0] <= width
    assert bbox[1] >= 0
    assert bbox[1] <= height
    assert bbox[2] >= 0
    assert bbox[2] <= width
    assert bbox[3] >= 0
    assert bbox[3] <= height


def test_generate_bbox_from_mask() -> None:
    """Test generate_bbox_from_mask."""
    gt_mask = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    width = 3
    height = 3

    bbox = generate_bbox_from_mask(gt_mask, width, height)

    assert isinstance(bbox, list)
    assert len(bbox) == 4
    assert bbox[0] >= 0
    assert bbox[0] <= width
    assert bbox[1] >= 0
    assert bbox[1] <= height
    assert bbox[2] >= 0
    assert bbox[2] <= width
    assert bbox[3] >= 0
    assert bbox[3] <= height