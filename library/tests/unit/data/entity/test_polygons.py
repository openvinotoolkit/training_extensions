# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the NaN-padded polygon carrier."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F  # noqa: N812

from getitune.data.entity.polygons import (
    clip_polygons,
    concat_polygons,
    empty_polygons,
    pad_polygons,
    polygon_lengths,
    polygons_from_object_array,
    polygons_to_object_array,
    scale_polygons,
)

CANVAS = (20, 20)


def test_nan_survives_resize() -> None:
    """NaN is the padding sentinel and must pass through transforms."""
    coords = torch.tensor([[[1.0, 1.0], [5.0, 5.0], [float("nan"), float("nan")]]])
    kp = tv_tensors.KeyPoints(coords, canvas_size=CANVAS)
    out = F.resize(kp, size=[40, 40])
    assert torch.isnan(out[0, 2]).all()
    assert torch.allclose(out[0, 0], torch.tensor([2.0, 2.0]))


def test_out_of_canvas_crop_behaviour() -> None:
    """pin whatever torchvision does to out-of-canvas vertices."""
    kp = tv_tensors.KeyPoints(torch.tensor([[[-10.0, -10.0], [5.0, 5.0]]]), canvas_size=CANVAS)
    out = F.crop(kp, top=0, left=0, height=10, width=10)
    assert out.shape == (1, 2, 2)
    assert torch.isfinite(out).all()


def test_empty_polygons_shape() -> None:
    poly = empty_polygons(CANVAS)
    assert poly.shape == (0, 0, 2)
    assert poly.dtype == torch.float32


def test_pad_polygons_ragged() -> None:
    rings = [torch.zeros((3, 2)), torch.ones((5, 2))]
    out = pad_polygons(rings, CANVAS)
    assert out.shape == (2, 5, 2)
    assert torch.isnan(out[0, 3:]).all()
    assert torch.isfinite(out[1]).all()
    assert polygon_lengths(out).tolist() == [3, 5]


def test_from_object_array_flat_and_nested() -> None:
    flat = np.array([0.0, 0.0, 4.0, 0.0, 4.0, 4.0], dtype=np.float32)
    nested = np.array([[0.0, 0.0], [8.0, 0.0], [8.0, 8.0], [0.0, 8.0]], dtype=np.float32)
    out = polygons_from_object_array([flat, nested], CANVAS)
    assert out.shape == (2, 4, 2)
    assert polygon_lengths(out).tolist() == [3, 4]


def test_instance_count_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="Expected 3 polygon entries"):
        polygons_from_object_array([np.zeros(6, dtype=np.float32)], CANVAS, num_instances=3)


def test_object_array_round_trip() -> None:
    src = [np.array([[1.0, 1.0], [5.0, 1.0], [5.0, 5.0]], dtype=np.float32)]
    out = polygons_to_object_array(polygons_from_object_array(src, CANVAS))
    np.testing.assert_allclose(out[0], src[0])


def test_multi_ring_separator() -> None:
    outer = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)
    hole = np.array([[3.0, 3.0], [7.0, 3.0], [7.0, 7.0], [3.0, 7.0]], dtype=np.float32)
    out = polygons_from_object_array([[outer, hole]], CANVAS)
    assert out.shape[1] == 9  # 4 + separator + 4
    assert torch.isnan(out[0, 4]).all()
    assert polygon_lengths(out).tolist() == [8]


def test_concat_pads_to_batch_pmax():
    a = pad_polygons([torch.zeros((3, 2))], CANVAS)  # (1, 3, 2)
    b = pad_polygons([torch.ones((7, 2))], CANVAS)  # (1, 7, 2)
    out = concat_polygons([a, b], CANVAS)
    assert out.shape == (2, 7, 2)
    assert polygon_lengths(out).tolist() == [3, 7]


def test_concat_handles_empty_and_none():
    a = pad_polygons([torch.zeros((4, 2))], CANVAS)
    out = concat_polygons([a, empty_polygons(CANVAS), None], CANVAS)
    assert out.shape == (1, 4, 2)
    assert concat_polygons([None, None], CANVAS).shape == (0, 0, 2)


def test_scale_preserves_nan_padding():
    src = pad_polygons([torch.ones((2, 2)), torch.ones((5, 2))], CANVAS)
    out = scale_polygons(src, 2.0, 3.0, 1.0, -1.0, canvas_size=(40, 60))
    assert torch.isnan(out[0, 2:]).all()
    assert torch.allclose(out[0, 0], torch.tensor([3.0, 2.0]))
    assert out.canvas_size == (40, 60)


def test_scale_empty_carrier_is_noop():
    out = scale_polygons(empty_polygons(CANVAS), 2.0, 2.0)
    assert out.shape == (0, 0, 2)


def test_clip_keeps_geometry_inside_rect():
    """The polygon analogue of the mosaic paste slice ``mask[:, y1:y2, x1:x2]``."""
    ring = torch.tensor([[-5.0, 5.0], [15.0, 5.0], [15.0, 15.0], [-5.0, 15.0]])
    out = clip_polygons(pad_polygons([ring], CANVAS), 0, 0, 10, 10, canvas_size=CANVAS)
    finite = out[0][torch.isfinite(out[0]).all(-1)]
    assert finite.tolist() == [[0.0, 5.0], [10.0, 5.0], [10.0, 10.0], [0.0, 10.0]]


def test_clip_grows_pmax_on_cut_corner():
    """A diagonal edge crossing a corner emits two intersections for one vertex,
    so ``P_max`` can exceed the input's -- callers must not assume it is stable."""
    tri = torch.tensor([[-2.0, 2.0], [6.0, -2.0], [6.0, 6.0]])
    out = clip_polygons(pad_polygons([tri], CANVAS), 0, 0, 10, 10, canvas_size=CANVAS)
    assert out.shape[1] == 5 > tri.shape[0]
    assert polygon_lengths(out).tolist() == [5]


def test_clip_fully_outside_becomes_all_nan_row():
    """Vanished instances stay as rows, matching the all-zero mask row the mask
    path produces -- dropping them would misalign polygons with bboxes/label."""
    inside = torch.tensor([[1.0, 1.0], [5.0, 1.0], [5.0, 5.0]])
    outside = torch.tensor([[50.0, 50.0], [60.0, 50.0], [60.0, 60.0]])
    out = clip_polygons(pad_polygons([inside, outside], CANVAS), 0, 0, 10, 10)
    assert out.shape[0] == 2
    assert torch.isnan(out[1]).all()
    assert polygon_lengths(out).tolist() == [3, 0]


def test_clip_fully_inside_is_identity():
    src = pad_polygons([torch.tensor([[1.0, 1.0], [5.0, 1.0], [5.0, 5.0]])], CANVAS)
    out = clip_polygons(src, 0, 0, 10, 10)
    torch.testing.assert_close(out.as_subclass(torch.Tensor), src.as_subclass(torch.Tensor), equal_nan=True)


def test_clip_preserves_ring_separator():
    """Each ring is clipped independently; losing the separator fills the hole."""
    outer = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)
    hole = np.array([[3.0, 3.0], [7.0, 3.0], [7.0, 7.0], [3.0, 7.0]], dtype=np.float32)
    src = polygons_from_object_array([[outer, hole]], CANVAS)
    out = clip_polygons(src, 0, 0, 10, 10)
    assert torch.isnan(out[0, 4]).all()
    assert polygon_lengths(out).tolist() == [8]


def test_clip_drops_degenerate_ring():
    """A ring reduced below 3 vertices is dropped, not emitted as a sliver."""
    sliver = torch.tensor([[-5.0, 5.0], [-1.0, 5.0], [-1.0, 8.0]])  # entirely left of x=0
    out = clip_polygons(pad_polygons([sliver], CANVAS), 0, 0, 10, 10)
    assert torch.isnan(out[0]).all()


def test_clip_empty_carrier_is_noop():
    out = clip_polygons(empty_polygons(CANVAS), 0, 0, 10, 10, canvas_size=(40, 40))
    assert out.shape == (0, 0, 2)
    assert out.canvas_size == (40, 40)


def test_clip_canvas_size_defaults_to_input():
    src = pad_polygons([torch.tensor([[1.0, 1.0], [5.0, 1.0], [5.0, 5.0]])], CANVAS)
    assert clip_polygons(src, 0, 0, 10, 10).canvas_size == CANVAS
