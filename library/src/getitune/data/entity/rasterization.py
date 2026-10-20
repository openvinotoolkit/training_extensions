# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Polygon → mask rasterization, with matched cv2 and torch backends.

Backend selection follows the tensor's device, not a capability import: CPU
tensors take the cv2 path, accelerator tensors stay on-device via the torch
path. ``rasterize_crop`` exists specifically so Mask R-CNN target generation
can run on-device and drop its host round-trip.
"""

from __future__ import annotations

import torch
from torchvision import tv_tensors

from getitune.data.entity.polygons import split_rings

__all__ = ["rasterize", "rasterize_batch", "rasterize_crop"]

#: Pixel-centre convention. A pixel ``i`` is sampled at continuous coordinate
#: ``i + 0.5``. cv2 samples on the integer grid, so cv2 input is offset by
#: ``-_CENTRE_OFFSET``; the torch scanline adds it. Both backends MUST agree or
#: they disagree by one pixel along every boundary.
_CENTRE_OFFSET: float = 0.5

#: cv2 fixed-point fractional bits. ``fillPoly`` takes integer vertices;
#: rounding to whole pixels destroys small crops (a 28x28 mask target has no
#: sub-pixel budget to lose), so coordinates are passed in 1/16-pixel units.
_FILL_SHIFT: int = 4
_FILL_SCALE: int = 1 << _FILL_SHIFT

#: Upper bound on ``edges * pixels`` per torch scanline chunk. The even-odd
#: test is O(E x H x W) materialised; without chunking a large instance on a
#: large canvas allocates tens of GB.
_MAX_CHUNK_ELEMENTS: int = 2**24


def _rasterize_cv2(coords: torch.Tensor, height: int, width: int, dtype: torch.dtype) -> torch.Tensor:
    import cv2
    import numpy as np

    n = coords.shape[0]
    out = np.zeros((n, height, width), dtype=np.uint8)
    if n and height and width:
        lim = (2**30) / _FILL_SCALE  # keep the fixed-point values inside int32
        c = coords.detach().to("cpu", torch.float64).clamp_(-lim, lim)
        for i, inst in enumerate(c):
            rings = split_rings(inst)
            if not rings:
                continue
            pts = [np.round((r.numpy() - _CENTRE_OFFSET) * _FILL_SCALE).astype(np.int32) for r in rings]
            # Every ring of an instance in ONE fillPoly call: cv2 applies the
            # even-odd rule across the polygons passed in a single call, so
            # per-ring calls fill holes solid instead of cutting them out.
            cv2.fillPoly(out[i], pts, color=1, lineType=cv2.LINE_8, shift=_FILL_SHIFT)
    return torch.from_numpy(out).to(device=coords.device, dtype=dtype)


def _outline_mask(ring: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Pixels cv2.fillPoly strokes along the ring's edges (LINE_8)."""
    dev = ring.device
    # same fixed-point -> integer pixel rounding cv2 applies before Line()
    q = torch.round((ring.double() - _CENTRE_OFFSET) * _FILL_SCALE)
    v = torch.floor((q + _FILL_SCALE / 2) / _FILL_SCALE)
    a, d = v, v.roll(-1, 0) - v
    steps = d.abs().amax(dim=1)  # (E,)
    cnt = steps.long() + 1
    eid = torch.repeat_interleave(torch.arange(len(a), device=dev), cnt)
    k = torch.arange(int(cnt.sum()), device=dev) - (torch.cumsum(cnt, 0) - cnt)[eid]
    t = k / steps[eid].clamp(min=1)
    x = torch.round(a[eid, 0] + t * d[eid, 0]).long()
    y = torch.round(a[eid, 1] + t * d[eid, 1]).long()
    ok = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    m = torch.zeros(height * width, dtype=torch.bool, device=dev)
    m[y[ok] * width + x[ok]] = True
    return m.view(height, width)


def _rasterize_torch(coords: torch.Tensor, height: int, width: int, dtype: torch.dtype) -> torch.Tensor:
    """Even-odd scanline fill, fully on-device and chunked over edges."""
    device = coords.device
    n = coords.shape[0]
    out = torch.zeros((n, height, width), dtype=torch.bool, device=device)
    if n == 0 or height == 0 or width == 0:
        return out.to(dtype)

    ys = torch.arange(height, device=device, dtype=coords.dtype) + _CENTRE_OFFSET
    xs = torch.arange(width, device=device, dtype=coords.dtype) + _CENTRE_OFFSET
    px = xs.view(1, 1, width)
    py = ys.view(1, height, 1)

    for i, inst in enumerate(coords):
        fill = torch.zeros((height, width), dtype=torch.bool, device=device)
        edge = torch.zeros_like(fill)
        for ring in split_rings(inst):
            p1 = ring
            p2 = torch.roll(ring, shifts=-1, dims=0)
            x1, y1 = p1[:, 0], p1[:, 1]
            x2, y2 = p2[:, 0], p2[:, 1]

            per_edge = max(1, height * width)
            chunk = max(1, min(ring.shape[0], _MAX_CHUNK_ELEMENTS // per_edge))
            acc = torch.zeros((height, width), dtype=torch.bool, device=device)
            for s in range(0, ring.shape[0], chunk):
                e = slice(s, s + chunk)
                ey1 = y1[e].view(-1, 1, 1)
                ey2 = y2[e].view(-1, 1, 1)
                ex1 = x1[e].view(-1, 1, 1)
                ex2 = x2[e].view(-1, 1, 1)

                straddles = (ey1 > py) != (ey2 > py)
                denom = torch.where(ey2 - ey1 == 0, torch.ones_like(ey2), ey2 - ey1)
                x_at = (ex2 - ex1) * (py - ey1) / denom + ex1
                crossings = straddles & (px < x_at)
                # XOR-accumulate: parity is what the even-odd rule needs, and
                # it makes holes fall out for free. ``acc`` is cumulative over
                # chunks, so it must be folded into ``fill`` only once, after
                # every edge of the ring has been counted.
                acc ^= ((crossings.to(torch.uint8).sum(0, dtype=torch.uint8)) & 1).bool()

            fill ^= acc
            edge |= _outline_mask(ring, height, width)
            out[i] = fill | edge

    return out.to(dtype)


def rasterize(
    coords: torch.Tensor,
    canvas_size: tuple[int, int] | None = None,
    dtype: torch.dtype = torch.uint8,
) -> tv_tensors.Mask:
    """Rasterize NaN-padded polygons to an ``(N, H, W)`` mask.

    Args:
        coords: ``(N, P_max, 2)`` polygons, typically a ``KeyPoints``.
        canvas_size: ``(H, W)``. Defaults to ``coords.canvas_size`` when
            ``coords`` carries one.
        dtype: Output dtype.

    Returns:
        ``tv_tensors.Mask`` of shape ``(N, H, W)``.

    Raises:
        ValueError: If no canvas size is available or ``coords`` is not 3-D.
    """
    if canvas_size is None:
        canvas_size = getattr(coords, "canvas_size", None)
    if canvas_size is None:
        msg = "canvas_size must be given when coords carries no canvas_size"
        raise ValueError(msg)
    if coords.ndim != 3 or coords.shape[-1] != 2:
        msg = f"Expected polygons of shape (N, P, 2), got {tuple(coords.shape)}"
        raise ValueError(msg)

    height, width = int(canvas_size[0]), int(canvas_size[1])
    plain = torch.as_tensor(coords.data if hasattr(coords, "data") else coords)

    if plain.shape[0] == 0:
        return tv_tensors.Mask(torch.zeros((0, height, width), dtype=dtype, device=plain.device))

    if plain.device.type == "cpu":
        return tv_tensors.Mask(_rasterize_cv2(plain, height, width, dtype))
    return tv_tensors.Mask(_rasterize_torch(plain, height, width, dtype))


def rasterize_crop(
    coords: torch.Tensor,
    boxes: torch.Tensor,
    out_size: int | tuple[int, int],
    inds: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Rasterize polygons directly into per-proposal crops.

    The polygon-native replacement for ``crop_and_resize_masks``: instead of
    rasterizing at full resolution and then ``roi_align``-ing down, each
    polygon is affinely mapped into its proposal's local frame and rasterized
    once at ``out_size``. This keeps everything on-device, which is what
    removes the host round-trip in Mask R-CNN target generation.

    Args:
        coords: ``(N, P_max, 2)`` polygons in image coordinates.
        boxes: ``(M, 4)`` proposals in XYXY image coordinates.
        out_size: Target crop size, ``int`` or ``(h, w)``.
        inds: ``(M,)`` index into ``coords`` giving each proposal's GT polygon.
        dtype: Output dtype.

    Returns:
        ``(M, out_h, out_w)`` tensor of crops.
    """
    out_h, out_w = (out_size, out_size) if isinstance(out_size, int) else out_size
    device = boxes.device
    m = boxes.shape[0]
    if m == 0 or coords.shape[0] == 0:
        return torch.zeros((m, out_h, out_w), dtype=dtype, device=device)

    plain = torch.as_tensor(coords.data if hasattr(coords, "data") else coords).to(device)
    sel = plain.index_select(0, inds.to(device=device, dtype=torch.long))

    x0 = boxes[:, 0].view(-1, 1)
    y0 = boxes[:, 1].view(-1, 1)
    # Degenerate proposals would divide by zero and poison the crop with inf.
    bw = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-4).view(-1, 1)
    bh = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-4).view(-1, 1)

    local = torch.empty_like(sel)
    local[..., 0] = (sel[..., 0] - x0) * (out_w / bw)
    local[..., 1] = (sel[..., 1] - y0) * (out_h / bh)

    if device.type == "cpu":
        return _rasterize_cv2(local, out_h, out_w, dtype)
    return _rasterize_torch(local, out_h, out_w, dtype)


def rasterize_batch(
    polygons: list[torch.Tensor],
    canvas_size: tuple[int, int],
    dtype: torch.dtype = torch.uint8,
) -> list[tv_tensors.Mask]:
    """Rasterize one batch of per-sample polygons at the network canvas size.

    Transitional bridge between the polygon-native data pipeline and the still
    mask-based model boundary. It runs *after* GPU augmentation, so polygons
    are already on-device and already at network resolution: this is the only
    rasterization left in the training path.

    Args:
        polygons: Per-sample ``(N_i, P_max, 2)`` NaN-padded polygons.
        canvas_size: ``(H, W)`` of the batch images.
        dtype: Mask dtype.

    Returns:
        One ``(N_i, H, W)`` mask per sample, in input order.
    """
    return [rasterize(p, canvas_size=canvas_size, dtype=dtype) for p in polygons]
