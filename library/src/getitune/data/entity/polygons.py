# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""NaN-padded polygon carrier for instance segmentation.

Polygons are stored as a ``tv_tensors.KeyPoints`` of shape ``(N, P_max, 2)``
in ``float32``, where ``N`` is the number of instances and ``P_max`` is the
largest vertex count in the sample.

NaN convention
--------------
* **Trailing** NaN rows are padding, so every instance can share ``P_max``.
* **Interior** NaN rows are ring separators, which is what makes multi-ring
  polygons (holes) representable without a layout migration later.

Using ``KeyPoints`` rather than a bare tensor is what lets torchvision v2
transforms resize/crop/flip polygons natively, with no transform-side changes.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch
from torchvision import tv_tensors

logger = logging.getLogger(__name__)

__all__ = [
    "empty_polygons",
    "pack_polygon_batch",
    "pad_polygons",
    "polygon_lengths",
    "polygons_from_object_array",
    "polygons_to_object_array",
    "split_rings",
    "unpack_polygon_batch",
]

#: Coordinate dtype. Matches ``bboxes`` in ``InstanceSegmentationSample``.
POLYGON_DTYPE: torch.dtype = torch.float32


def split_rings(inst: torch.Tensor) -> list[torch.Tensor]:
    """Split one instance's ``(P, 2)`` rows into rings on NaN separators."""
    valid = torch.isfinite(inst).all(dim=-1)
    rings: list[torch.Tensor] = []
    start: int | None = None
    for i, ok in enumerate(valid.tolist()):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start >= 3:
                rings.append(inst[start:i])
            start = None
    if start is not None and inst.shape[0] - start >= 3:
        rings.append(inst[start:])
    return rings


def empty_polygons(canvas_size: tuple[int, int]) -> tv_tensors.KeyPoints:
    """Return the canonical empty polygon set, shape ``(0, 0, 2)``.

    Mirrors how ``InstanceSegmentationSample.__post_init__`` normalises a
    missing ``masks`` field to ``Mask(torch.zeros((0, *shape)))``. Every
    rasterization and validation path must survive this value — the
    no-annotation image is the case that breaks in production, not the happy
    path.
    """
    return tv_tensors.KeyPoints(
        torch.zeros((0, 0, 2), dtype=POLYGON_DTYPE),
        canvas_size=canvas_size,
    )


def _ring_stack(entry: object) -> torch.Tensor:
    """Normalise one instance's annotation into a ``(P, 2)`` tensor.

    Accepts a single ring as ``(P, 2)`` or flat ``(2P,)``, or a sequence of
    rings, which are concatenated with a single NaN separator row between them.
    """
    if isinstance(entry, (list, tuple)) and len(entry) and not np.isscalar(entry[0]):
        rings = [_ring_stack(r) for r in entry]
    else:
        arr = np.asarray(entry, dtype=np.float32)
        rings = [torch.from_numpy(arr.reshape(-1, 2))]

    if len(rings) == 1:
        return rings[0]

    sep = torch.full((1, 2), float("nan"), dtype=POLYGON_DTYPE)
    out: list[torch.Tensor] = []
    for i, ring in enumerate(rings):
        if i:
            out.append(sep)
        out.append(ring)
    return torch.cat(out, dim=0)


def pad_polygons(
    coords: Sequence[torch.Tensor],
    canvas_size: tuple[int, int],
    p_max: int | None = None,
) -> tv_tensors.KeyPoints:
    """Pad a ragged list of ``(P_i, 2)`` rings into ``(N, P_max, 2)``.

    Args:
        coords: One ``(P_i, 2)`` float tensor per instance.
        canvas_size: ``(H, W)`` of the image the coordinates refer to.
        p_max: Target vertex count. Defaults to the largest ``P_i``. Padding is
            deliberately **per sample**; per-batch padding is a later concern
            and pre-solving it here would couple this module to collation.

    Returns:
        NaN-padded ``KeyPoints`` of shape ``(N, P_max, 2)``.
    """
    if not len(coords):
        return empty_polygons(canvas_size)

    lengths = [c.shape[0] for c in coords]
    target = int(p_max if p_max is not None else max(lengths))

    out = torch.full((len(coords), target, 2), float("nan"), dtype=POLYGON_DTYPE)
    for i, ring in enumerate(coords):
        n = min(ring.shape[0], target)
        if n < ring.shape[0]:
            logger.warning("Polygon %d truncated from %d to %d vertices", i, ring.shape[0], target)
        out[i, :n] = ring[:n].to(POLYGON_DTYPE)

    return tv_tensors.KeyPoints(out, canvas_size=canvas_size)


def polygons_from_object_array(
    array: Sequence[object] | np.ndarray,
    canvas_size: tuple[int, int],
    num_instances: int | None = None,
) -> tv_tensors.KeyPoints:
    """Build a padded polygon carrier from a ragged per-instance array.

    Args:
        array: One entry per instance, each a ``(P, 2)`` / flat ``(2P,)`` array
            or a sequence of such rings. This is the layout already produced by
            Datumaro and consumed by ``mask_util.polygon_to_bitmap``.
        canvas_size: ``(H, W)`` of the source image.
        num_instances: Expected instance count, for cross-checking against
            ``bboxes`` / ``label``. A mismatch is an ingestion bug and raises
            rather than silently misaligning polygons to labels.

    Returns:
        NaN-padded ``KeyPoints`` of shape ``(N, P_max, 2)``.

    Raises:
        ValueError: If ``num_instances`` disagrees with ``len(array)``.
    """
    entries = list(array) if array is not None else []
    if num_instances is not None and len(entries) != num_instances:
        msg = f"Expected {num_instances} polygon entries, got {len(entries)}"
        raise ValueError(msg)
    if not entries:
        return empty_polygons(canvas_size)
    return pad_polygons([_ring_stack(e) for e in entries], canvas_size)


def polygon_lengths(coords: torch.Tensor) -> torch.Tensor:
    """Return the per-instance count of non-padding vertex rows.

    A row counts as real when **both** coordinates are finite, so ring
    separators and trailing padding are excluded alike.
    """
    if coords.numel() == 0:
        return torch.zeros((coords.shape[0],), dtype=torch.long, device=coords.device)
    return torch.isfinite(coords).all(dim=-1).sum(dim=-1)


def polygons_to_object_array(coords: torch.Tensor) -> list[np.ndarray]:
    """Strip padding and return one ``(P_i, 2)`` float32 array per instance.

    Inverse of :func:`polygons_from_object_array` for single-ring polygons.
    Ring separators are dropped, so a multi-ring instance round-trips as a
    single concatenated ring — sufficient for the pycocotools oracle, not for
    hole-preserving export.
    """
    out: list[np.ndarray] = []
    for inst in coords:
        valid = torch.isfinite(inst).all(dim=-1)
        out.append(inst[valid].detach().cpu().numpy().astype(np.float32))
    return out


def _plain(coords: torch.Tensor) -> torch.Tensor:
    """Unwrap a ``KeyPoints`` to its backing tensor."""
    return torch.as_tensor(coords.data if hasattr(coords, "data") else coords)


def concat_polygons(
    items: Sequence[torch.Tensor | None],
    canvas_size: tuple[int, int],
) -> tv_tensors.KeyPoints:
    """Concatenate per-sample ``(N_i, P_i, 2)`` carriers into ``(sum N_i, P_max, 2)``.

    ``P_max`` differs per sample, so every entry is NaN-padded to the batch-wide
    maximum before the ``cat``. This is the polygon analogue of stacking masks
    on a common canvas — without it mosaic/mixup concatenation raises.
    """
    tensors = [_plain(t) for t in items if t is not None]
    if not tensors:
        return empty_polygons(canvas_size)

    p_max = max(int(t.shape[1]) for t in tensors)
    padded: list[torch.Tensor] = []
    for t in tensors:
        if t.shape[0] == 0:
            padded.append(t.reshape(0, p_max, 2))
            continue
        if t.shape[1] < p_max:
            pad = torch.full(
                (t.shape[0], p_max - t.shape[1], 2),
                float("nan"),
                dtype=t.dtype,
                device=t.device,
            )
            t = torch.cat([t, pad], dim=1)  # noqa: PLW2901
        padded.append(t)
    return tv_tensors.KeyPoints(torch.cat(padded, dim=0), canvas_size=canvas_size)


def scale_polygons(
    coords: torch.Tensor,
    scale_x: float,
    scale_y: float,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    canvas_size: tuple[int, int] | None = None,
) -> tv_tensors.KeyPoints:
    """Affine-scale + translate polygon coordinates. NaN padding is preserved."""
    plain = _plain(coords).clone().to(POLYGON_DTYPE)
    if plain.numel():
        plain[..., 0] = plain[..., 0] * scale_x + offset_x
        plain[..., 1] = plain[..., 1] * scale_y + offset_y
    target = canvas_size if canvas_size is not None else getattr(coords, "canvas_size", (0, 0))
    return tv_tensors.KeyPoints(plain, canvas_size=target)


#: Below this the clip plane is parallel to the edge and the crossing test
#: already excluded it; the value only keeps the division finite.
_CLIP_EPS: float = 1e-12


def _clip_ring(ring: torch.Tensor, x1: float, y1: float, x2: float, y2: float) -> torch.Tensor:
    """Sutherland-Hodgman clip of one ``(P, 2)`` ring to an axis-aligned rect.

    The rect is convex, so the four half-plane passes compose without the
    self-intersection artifacts a general polygon-polygon clip would need to
    handle. Vertex count can grow (a cut corner adds one) or drop to zero.
    """
    for axis, bound, keep_above in ((0, x1, True), (0, x2, False), (1, y1, True), (1, y2, False)):
        if ring.shape[0] < 3:
            return ring.new_zeros((0, 2))

        coord = ring[:, axis]
        inside = coord >= bound if keep_above else coord <= bound
        nxt_inside = inside.roll(-1, 0)

        p = ring
        q = ring.roll(-1, 0)
        denom = q[:, axis] - p[:, axis]
        # A crossing implies denom != 0; the guard only stops 0/0 -> NaN on the
        # non-crossing edges, whose t is discarded anyway.
        safe = torch.where(denom.abs() < _CLIP_EPS, torch.ones_like(denom), denom)
        t = ((bound - p[:, axis]) / safe).unsqueeze(-1)
        inter = p + t * (q - p)
        inter[:, axis] = bound  # pin exactly on the plane, no float drift

        # Interleave [vertex, intersection] per edge so the output keeps
        # winding order; selecting with a flat mask is the vectorised form of
        # the classic "emit p, then emit the crossing" loop.
        cand = torch.stack([p, inter], dim=1).reshape(-1, 2)
        keep = torch.stack([inside, inside ^ nxt_inside], dim=1).reshape(-1)
        ring = cand[keep]

    return ring if ring.shape[0] >= 3 else ring.new_zeros((0, 2))


def clip_polygons(
    coords: torch.Tensor,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    canvas_size: tuple[int, int] | None = None,
) -> tv_tensors.KeyPoints:
    """Clip polygons to ``[x1, x2] x [y1, y2]``, discarding outside geometry.

    This is the polygon analogue of the mosaic paste slice
    ``mask[:, y1:y2, x1:x2]``.

    Instances whose rings all vanish become all-NaN rows, which rasterize to an
    empty mask -- matching the all-zero mask row the mask path produces, so
    instance indices stay aligned with ``bboxes`` and ``label``.

    Args:
        coords: ``(N, P_max, 2)`` NaN-padded polygons.
        x1: Left edge of the clip rect.
        y1: Top edge of the clip rect.
        x2: Right edge of the clip rect.
        y2: Bottom edge of the clip rect.
        canvas_size: ``(H, W)`` for the result. Defaults to the input's.

    Returns:
        NaN-padded ``KeyPoints``; ``P_max`` may differ from the input's.
    """
    target = canvas_size if canvas_size is not None else getattr(coords, "canvas_size", (0, 0))
    plain = _plain(coords).to(POLYGON_DTYPE)
    if plain.numel() == 0:
        return tv_tensors.KeyPoints(plain.clone(), canvas_size=target)

    sep = torch.full((1, 2), float("nan"), dtype=POLYGON_DTYPE, device=plain.device)
    clipped: list[torch.Tensor] = []
    for inst in plain:
        rings = [_clip_ring(r, x1, y1, x2, y2) for r in split_rings(inst)]
        rings = [r for r in rings if r.shape[0] >= 3]
        if not rings:
            clipped.append(plain.new_full((0, 2), float("nan")))
            continue
        parts: list[torch.Tensor] = []
        for i, ring in enumerate(rings):
            if i:
                parts.append(sep)
            parts.append(ring)
        clipped.append(torch.cat(parts, dim=0))

    p_max = max(1, *(r.shape[0] for r in clipped))
    return pad_polygons(clipped, tuple(target), p_max=p_max)


def pack_polygon_batch(
    items: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Flatten ragged per-sample polygons into the ``(B, M, 2)`` Kornia wants.

    Kornia's keypoint path takes one stacked tensor, but ``N_i`` (instances)
    and ``P_i`` (vertices) both vary per sample. Each sample is flattened to
    ``(N_i * P_i, 2)`` and NaN-padded to the batch-wide maximum. NaN is already
    the padding sentinel, and an affine transform maps NaN to NaN, so the
    padding survives the round trip without a separate validity mask.

    Args:
        items: One ``(N_i, P_i, 2)`` carrier per sample.

    Returns:
        ``(packed, shapes)`` where ``packed`` is ``(B, M, 2)`` and ``shapes``
        holds the original ``(N_i, P_i)`` needed to invert this.
    """
    shapes = [(int(t.shape[0]), int(t.shape[1])) for t in items]
    flat = [_plain(t).reshape(-1, 2) for t in items]
    m = max((f.shape[0] for f in flat), default=0)

    ref = flat[0] if flat else torch.zeros((0, 2), dtype=POLYGON_DTYPE)
    packed = torch.full((len(flat), m, 2), float("nan"), dtype=POLYGON_DTYPE, device=ref.device)
    for i, f in enumerate(flat):
        if f.shape[0]:
            packed[i, : f.shape[0]] = f.to(POLYGON_DTYPE)
    return packed, shapes


def unpack_polygon_batch(
    packed: torch.Tensor,
    shapes: Sequence[tuple[int, int]],
    canvas_size: tuple[int, int],
) -> list[tv_tensors.KeyPoints]:
    """Invert :func:`pack_polygon_batch`, restoring per-sample ``(N_i, P_i, 2)``.

    Args:
        packed: ``(B, M, 2)`` transformed coordinates.
        shapes: Per-sample ``(N_i, P_i)`` from the matching pack call.
        canvas_size: ``(H, W)`` of the *transformed* images.

    Returns:
        One ``KeyPoints`` per sample, in input order.

    Raises:
        ValueError: If ``packed`` and ``shapes`` disagree on the batch size.
    """
    if packed.shape[0] != len(shapes):
        msg = f"Packed batch {packed.shape[0]} does not match {len(shapes)} shapes"
        raise ValueError(msg)

    out: list[tv_tensors.KeyPoints] = []
    for row, (n, p) in zip(packed, shapes):
        coords = row[: n * p].reshape(n, p, 2).to(POLYGON_DTYPE)
        out.append(tv_tensors.KeyPoints(coords, canvas_size=canvas_size))
    return out
