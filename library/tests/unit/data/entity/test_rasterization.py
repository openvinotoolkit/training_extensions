# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for polygon rasterization: backend parity and fill-rule fixes."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from getitune.data.entity.polygons import polygons_from_object_array
from getitune.data.entity.rasterization import (
    _rasterize_cv2,
    _rasterize_torch,
    rasterize,
    rasterize_crop,
)
from getitune.data.utils.structures.mask.mask_util import polygon_to_bitmap

CANVAS = (64, 64)


def _square(x0: float, y0: float, size: float) -> np.ndarray:
    return np.array(
        [[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size]],
        dtype=np.float32,
    )


def _iou(a: torch.Tensor, b: torch.Tensor) -> float:
    a_b, b_b = a.bool(), b.bool()
    union = (a_b | b_b).sum().item()
    return 1.0 if union == 0 else (a_b & b_b).sum().item() / union


@pytest.mark.parametrize(
    "poly",
    [_square(8, 8, 20), _square(0.5, 0.5, 13.25), _square(-5, -5, 30)],
    ids=["aligned", "subpixel", "clipped"],
)
def test_cv2_torch_parity(poly: np.ndarray) -> None:
    coords = polygons_from_object_array([poly], CANVAS)
    plain = torch.as_tensor(coords.data)
    cv = _rasterize_cv2(plain, *CANVAS, torch.uint8)
    th = _rasterize_torch(plain, *CANVAS, torch.uint8)
    assert _iou(cv, th) >= 0.99


def test_parity_survives_erosion() -> None:
    """IoU alone is blind to the 1px boundary shift _CENTRE_OFFSET prevents.

    Eroding both masks strips the shared interior, so any systematic
    half-pixel disagreement between backends dominates what is left.
    """
    poly: np.ndarray = _square(8, 8, 20)
    coords = torch.as_tensor(polygons_from_object_array([poly], CANVAS).data)
    cv = _rasterize_cv2(coords, *CANVAS, torch.uint8)
    th = _rasterize_torch(coords, *CANVAS, torch.uint8)
    xor = (cv.bool() ^ th.bool()).sum().item()
    assert xor <= 2, f"backends disagree on {xor} pixels"


def test_pycocotools_oracle() -> None:
    """Our rasterization vs pycocotools, modulo the known boundary convention.

    ``_rasterize_cv2`` applies ``_CENTRE_OFFSET`` and inherits ``fillPoly``'s
    inclusive LINE_8 outline; pycocotools' ``rleFrPoly`` does neither. The
    result is a one-pixel shell around an otherwise identical interior.
    """
    poly = _square(10, 10, 70)
    coords = torch.as_tensor(polygons_from_object_array([poly], CANVAS).data)
    ours = _rasterize_cv2(coords, *CANVAS, torch.uint8)
    oracle = torch.from_numpy(polygon_to_bitmap(np.array([poly], dtype=object), *CANVAS))
    assert _iou(ours, oracle.to(torch.uint8)) >= 0.99


def test_hole_is_cut_out() -> None:
    """All rings in one fillPoly call: per-ring calls fill the hole solid."""
    outer = _square(8, 8, 40)
    hole = _square(20, 20, 16)
    coords = torch.as_tensor(polygons_from_object_array([[outer, hole]], CANVAS).data)
    for backend in (_rasterize_cv2, _rasterize_torch):
        mask = backend(coords, *CANVAS, torch.uint8)
        assert mask[0, 28, 28] == 0, f"{backend.__name__} filled the hole"
        assert mask[0, 10, 10] == 1


def test_subpixel_survives_small_crop() -> None:
    """Fixed-point shift: rounding to int destroys 28x28 mask targets."""
    poly = _square(0.4, 0.4, 3.2)
    coords = torch.as_tensor(polygons_from_object_array([poly], (8, 8)).data)
    mask = _rasterize_cv2(coords, 8, 8, torch.uint8)
    assert 0 < mask.sum().item() < 64


def test_chunking_matches_unchunked(monkeypatch: pytest.MonkeyPatch) -> None:
    poly = np.stack(
        [
            20 + 15 * np.cos(np.linspace(0, 2 * np.pi, 200, endpoint=False)),
            20 + 15 * np.sin(np.linspace(0, 2 * np.pi, 200, endpoint=False)),
        ],
        axis=1,
    ).astype(np.float32)
    coords = torch.as_tensor(polygons_from_object_array([poly], CANVAS).data)
    full = _rasterize_torch(coords, *CANVAS, torch.uint8)
    monkeypatch.setattr("getitune.data.entity.rasterization._MAX_CHUNK_ELEMENTS", 64 * 64 * 4)
    chunked = _rasterize_torch(coords, *CANVAS, torch.uint8)
    assert torch.equal(full, chunked)


@pytest.mark.parametrize(
    "poly",
    [np.zeros((0, 2), np.float32), _square(-100, -100, 5)],
    ids=["degenerate", "off_canvas"],
)
def test_degenerate_inputs_do_not_raise(poly: np.ndarray) -> None:
    coords = polygons_from_object_array([poly], CANVAS)
    assert rasterize(coords, CANVAS).shape == (1, *CANVAS)


def test_empty_instance_set() -> None:
    from getitune.data.entity.polygons import empty_polygons

    assert rasterize(empty_polygons(CANVAS), CANVAS).shape == (0, *CANVAS)


def test_rasterize_crop_matches_full_res_crop() -> None:
    """rasterize_crop must agree with rasterize-then-crop, not just run."""
    poly = _square(16, 16, 32)
    coords = polygons_from_object_array([poly], CANVAS)
    boxes = torch.tensor([[16.0, 16.0, 48.0, 48.0]])
    crop = rasterize_crop(coords, boxes, 28, torch.zeros(1, dtype=torch.long))
    assert crop.shape == (1, 28, 28)
    assert crop.mean().item() > 0.9  # box tightly bounds the square


def test_rasterize_crop_empty_proposals() -> None:
    coords = polygons_from_object_array([_square(4, 4, 8)], CANVAS)
    out = rasterize_crop(coords, torch.zeros((0, 4)), 28, torch.zeros(0, dtype=torch.long))
    assert out.shape == (0, 28, 28)
