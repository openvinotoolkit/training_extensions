# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from datumaro.experimental import Dataset
from datumaro.experimental.fields import ImageInfo as DmImageInfo
from datumaro.experimental.fields import Subset
from torchvision import tv_tensors

from getitune.data.entity import empty_polygons
from getitune.data.entity.polygons import polygons_from_object_array
from getitune.data.entity.rasterization import rasterize
from getitune.data.entity.sample import (
    InstanceSegmentationPolygonSample,
    InstanceSegmentationSample,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


#: Ragged on purpose: a triangle, a square, and a square-with-hole.
#: P_i = 3 / 4 / 9 forces every NaN-padding and ring-separator path.
_RINGS: list = [
    np.array([[4.0, 4.0], [28.0, 6.0], [10.0, 30.0]], dtype=np.float32),
    np.array([[40.0, 40.0], [80.0, 40.0], [80.0, 80.0], [40.0, 80.0]], dtype=np.float32),
    [
        np.array([[10.0, 50.0], [34.0, 50.0], [34.0, 90.0], [10.0, 90.0]], dtype=np.float32),
        np.array([[18.0, 60.0], [26.0, 60.0], [26.0, 80.0], [18.0, 80.0]], dtype=np.float32),
    ],
]


def _bboxes_from(coords: torch.Tensor, canvas_size: tuple[int, int]) -> tv_tensors.BoundingBoxes:
    """Tight XYXY boxes over the finite vertices of each instance."""
    boxes = []
    for inst in coords:
        valid = inst[torch.isfinite(inst).all(dim=-1)]
        boxes.append(torch.tensor([valid[:, 0].min(), valid[:, 1].min(), valid[:, 0].max(), valid[:, 1].max()]))
    data = torch.stack(boxes) if boxes else torch.zeros((0, 4))
    return tv_tensors.BoundingBoxes(data.float(), format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=canvas_size)


def _scale_entry(entry: object, scale_x: float, scale_y: float) -> object:
    """Scale one instance's rings by (scale_x, scale_y).

    Mirrors the layout `_ring_stack` accepts: either a single (P, 2) array or a
    sequence of such arrays for a multi-ring instance.
    """
    if isinstance(entry, (list, tuple)):
        return [_scale_entry(r, scale_x, scale_y) for r in entry]
    arr = np.asarray(entry, dtype=np.float32).reshape(-1, 2).copy()
    arr[:, 0] *= scale_x
    arr[:, 1] *= scale_y
    return arr


def _entries_for(hw: tuple[int, int], rings: list) -> list:
    """Rings scaled from the 100x100 reference canvas to `hw`."""
    sx, sy = hw[1] / 100.0, hw[0] / 100.0
    return [_scale_entry(entry, sx, sy) for entry in rings]


@pytest.fixture
def fxt_polygon_rings() -> list:
    """Deterministic ragged rings, including one multi-ring (hole) instance."""
    return _RINGS


@pytest.fixture
def fxt_polygon_sample(fxt_polygon_rings):
    """Factory for a polygon-native sample (no ``masks`` field)."""

    def _make(
        hw: tuple[int, int] = (100, 100), rings: list | None = None, seed: int = 0
    ) -> InstanceSegmentationPolygonSample:
        torch.manual_seed(seed)
        entries = _entries_for(hw, fxt_polygon_rings if rings is None else rings)
        coords = polygons_from_object_array(entries, hw)
        return InstanceSegmentationPolygonSample(
            subset=Subset.TRAINING,
            image=tv_tensors.Image(torch.rand(3, *hw)),
            dm_image_info=DmImageInfo(height=hw[0], width=hw[1]),
            bboxes=_bboxes_from(coords, hw),
            polygons=coords,
            label=torch.arange(coords.shape[0], dtype=torch.uint8),
        )

    return _make


@pytest.fixture
def fxt_mask_sample(fxt_polygon_rings):
    """Factory for the mask-schema twin: identical geometry, rasterized up front.

    The image tensor is generated from the same seed as ``fxt_polygon_sample``
    so seeded transforms (mosaic/mixup) take identical random paths and the
    two can be compared pixel-for-pixel.
    """

    def _make(hw: tuple[int, int] = (100, 100), rings: list | None = None, seed: int = 0) -> InstanceSegmentationSample:
        torch.manual_seed(seed)
        entries = _entries_for(hw, fxt_polygon_rings if rings is None else rings)
        coords = polygons_from_object_array(entries, hw)
        return InstanceSegmentationSample(
            subset=Subset.TRAINING,
            image=tv_tensors.Image(torch.rand(3, *hw)),
            dm_image_info=DmImageInfo(height=hw[0], width=hw[1]),
            bboxes=_bboxes_from(coords, hw),
            masks=rasterize(coords, canvas_size=hw),
            polygons=empty_polygons(hw),
            label=torch.arange(coords.shape[0], dtype=torch.uint8),
        )

    return _make


@pytest.fixture
def fxt_mock_classification_dm_subset(mocker: MockerFixture) -> MagicMock:
    mock_dm_subset = mocker.MagicMock(spec=Dataset)
    mock_dm_subset.__len__.return_value = 1
    return mock_dm_subset


@pytest.fixture
def fxt_mock_detection_dm_subset(mocker: MockerFixture) -> MagicMock:
    mock_dm_subset = mocker.MagicMock(spec=Dataset)
    mock_dm_subset.__len__.return_value = 1
    return mock_dm_subset


@pytest.fixture
def fxt_mock_segmentation_dm_subset(mocker: MockerFixture) -> MagicMock:
    mock_dm_subset = mocker.MagicMock(spec=Dataset)
    mock_dm_subset.__len__.return_value = 1
    return mock_dm_subset


@pytest.fixture
def fxt_mock_instance_seg_dm_subset(mocker):
    """Mock subset factory; ``has_polygons`` drives _resolve_annotation_format."""

    def _make(has_polygons: bool = True) -> MagicMock:
        subset = mocker.MagicMock(spec=Dataset)
        subset.__len__.return_value = 2
        subset.convert_to_schema.return_value = subset
        attrs = {"label": MagicMock(), "bboxes": MagicMock()}
        if has_polygons:
            attrs["polygons"] = MagicMock()
        else:
            attrs["masks"] = MagicMock()
        subset.schema = MagicMock(attributes=attrs)
        subset.label_categories = MagicMock(labels=["a", "b", "c"])
        return subset

    return _make
