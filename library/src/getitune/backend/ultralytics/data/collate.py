# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom collate function producing Ultralytics batch dicts."""

from __future__ import annotations

from typing import Any, Callable

import torch

from getitune.data.entity.polygons import split_rings
from getitune.data.entity.rasterization import rasterize

#: Ultralytics proto-head downsample. Masks are consumed at ``imgsz / 4``, so
#: that is the only resolution worth rasterizing at — anything larger is built
#: and immediately thrown away by ``SegmentationTrainer.preprocess_batch``.
MASK_RATIO: int = 4


def _polygon_areas(coords: torch.Tensor) -> torch.Tensor:
    """Return the per-instance absolute shoelace area of ``(N, P_max, 2)`` polygons.

    Used only for the overlap sort order, which is why ring areas are summed
    rather than signed-combined: an exact hole-aware area is not needed to
    decide which instance should win an overlapping pixel, and computing it
    from geometry avoids rasterizing just to call ``.sum()``.
    """
    areas = torch.zeros(coords.shape[0], dtype=torch.float32)
    for i, inst in enumerate(coords):
        total = 0.0
        for ring in split_rings(inst):
            x, y = ring[:, 0], ring[:, 1]
            total += float(torch.abs(torch.dot(x, y.roll(-1)) - torch.dot(y, x.roll(-1))) / 2)
        areas[i] = total
    return areas


def _build_index_and_sem(masks: torch.Tensor, cls_i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the ``(H, W)`` overlap index map and semantic map from sorted masks.

    ``masks`` must already be sorted by descending area, matching upstream
    Ultralytics ``overlap_mask=True``: instance ``k`` is painted with value
    ``k + 1``, so the *last* (smallest) instance wins a contested pixel.
    Taking the per-pixel max over ``mask_k * (k + 1)`` is the vectorised
    equivalent of painting them in order.

    Args:
        masks: ``(N, H, W)`` binary instance masks, area-descending.
        cls_i: ``(N, 1)`` class ids in the same order.

    Returns:
        ``(index_map, sem_map)``, both ``(H, W)``.
    """
    n, h, w = masks.shape
    if n == 0:
        return torch.zeros((h, w), dtype=torch.uint8), torch.zeros((h, w), dtype=torch.float32)

    idx = torch.arange(1, n + 1, dtype=torch.float32).view(-1, 1, 1)
    index_map = ((masks > 0).float() * idx).amax(0).to(torch.uint8)

    sem = torch.zeros((h, w), dtype=torch.float32)
    fg = index_map > 0
    if fg.any():
        sem[fg] = cls_i.squeeze(-1)[index_map[fg].long() - 1]
    return index_map, sem


def _stack_geometry_fields(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collect the per-image geometry/metadata fields shared by all collate functions."""
    return {
        "ori_shape": [b["ori_shape"] for b in batch],
        "resized_shape": [b["resized_shape"] for b in batch],
        "ratio_pad": [b["ratio_pad"] for b in batch],
        "im_file": [b.get("im_file", "") for b in batch],
    }


def _collect_detection_targets(
    batch: list[dict[str, Any]],
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Build per-image ``cls``/``bboxes``/``batch_idx`` tensors from adapter samples."""
    all_cls: list[torch.Tensor] = []
    all_bboxes: list[torch.Tensor] = []
    all_batch_idx: list[torch.Tensor] = []

    for i, b in enumerate(batch):
        cls = torch.as_tensor(b["cls"], dtype=torch.float32)
        bboxes = torch.as_tensor(b["bboxes"], dtype=torch.float32)
        n = cls.shape[0]
        all_cls.append(cls)
        all_bboxes.append(bboxes)
        all_batch_idx.append(torch.full((n,), i, dtype=torch.float32))

    return all_cls, all_bboxes, all_batch_idx


def _base_collate(
    batch: list[dict[str, Any]],
    target_key: str,
    collate_target_fn: Callable[[list[Any]], Any],
) -> dict[str, Any]:
    """Collate shared metadata fields and apply a custom collator to the target key.

    Args:
        batch: List of adapter sample dicts.
        target_key: Key holding the task-specific label/mask (e.g. ``"cls"``).
        collate_target_fn: Function that stacks/converts the per-sample values
            of ``target_key`` into a single batched tensor.
    """
    return {
        "img": torch.stack([b["img"] for b in batch], dim=0),
        target_key: collate_target_fn([b[target_key] for b in batch]),
        **_stack_geometry_fields(batch),
    }


def detection_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate detection adapter dicts into an Ultralytics-compatible batch.

    Stacks images into ``(B, C, H, W)`` and concatenates per-sample
    annotations with correct per-image ``batch_idx``.
    """
    all_cls, all_bboxes, all_batch_idx = _collect_detection_targets(batch)

    return {
        "img": torch.stack([b["img"] for b in batch], dim=0),
        "cls": torch.cat(all_cls, dim=0) if all_cls else torch.zeros((0, 1), dtype=torch.float32),
        "bboxes": torch.cat(all_bboxes, dim=0) if all_bboxes else torch.zeros((0, 4), dtype=torch.float32),
        "batch_idx": torch.cat(all_batch_idx, dim=0) if all_batch_idx else torch.zeros(0, dtype=torch.float32),
        **_stack_geometry_fields(batch),
    }


def instance_seg_collate_fn(batch: list[dict[str, Any]], mask_ratio: int = MASK_RATIO) -> dict[str, Any]:
    """Collate instance-segmentation adapter dicts into an Ultralytics-compatible batch.

    Accepts either annotation carrier produced by
    :meth:`UltralyticsDatasetAdapter._getitem_segment`:

    * ``segments`` — normalised polygons.  Rasterized here exactly once, at
      proto resolution ``imgsz / mask_ratio``, which is the resolution the
      segmentation loss consumes.  No full-resolution mask is ever built.
    * ``masks`` — dense fallback, rasterized upstream at full resolution.

    Both paths emit the same contract: ``masks`` as a ``(B, H, W)`` uint8
    overlap index map (values ``1..N``, ``0`` = background) with instances
    sorted by descending area, plus the matching ``(B, H, W)`` ``sem_masks``.
    ``cls`` and ``bboxes`` are reordered to that sort order.

    Args:
        batch: Adapter sample dicts.
        mask_ratio: Proto-head downsample applied on the polygon path.

    Returns:
        Ultralytics batch dict.

    Raises:
        ValueError: If a batch mixes polygon and dense-mask samples that both
            carry instances — the two would land at different resolutions and
            could not be stacked.
    """
    all_cls, all_bboxes, all_batch_idx = _collect_detection_targets(batch)

    use_polygons = any("segments" in b for b in batch)
    img_h, img_w = batch[0]["img"].shape[-2:]
    out_h, out_w = (img_h // mask_ratio, img_w // mask_ratio) if use_polygons else (img_h, img_w)

    overlap_maps: list[torch.Tensor] = []
    sem_maps: list[torch.Tensor] = []

    for i, b in enumerate(batch):
        segments = b.get("segments")

        if segments is None:
            masks = b.get("masks")
            if masks is None or masks.shape[0] == 0:
                # No instances: the adapter has no geometry to emit, so this
                # sample legitimately has no carrier. Emit at the batch's
                # resolution so the stack below is homogeneous.
                overlap_maps.append(torch.zeros((out_h, out_w), dtype=torch.uint8))
                sem_maps.append(torch.zeros((out_h, out_w), dtype=torch.float32))
                continue
            if use_polygons:
                msg = (
                    "Batch mixes polygon and dense-mask instance annotations; "
                    "annotation_format must be consistent across a subset."
                )
                raise ValueError(msg)
            if not isinstance(masks, torch.Tensor):
                masks = torch.as_tensor(masks, dtype=torch.float32)
            areas = masks.float().sum(dim=(1, 2))
        else:
            # Normalised -> proto-resolution pixel coords. NaN padding and ring
            # separators survive the scale and are consumed by ``rasterize``.
            coords = segments.clone()
            coords[..., 0] *= out_w
            coords[..., 1] *= out_h
            areas = _polygon_areas(coords)
            masks = None

        sorted_idx = torch.argsort(areas, descending=True)
        all_cls[i] = all_cls[i][sorted_idx]
        all_bboxes[i] = all_bboxes[i][sorted_idx]

        if segments is None:
            sorted_masks = masks[sorted_idx]
        else:
            sorted_masks = rasterize(coords[sorted_idx], canvas_size=(out_h, out_w))

        index_map, sem = _build_index_and_sem(sorted_masks, all_cls[i])
        overlap_maps.append(index_map)
        sem_maps.append(sem)

    return {
        "img": torch.stack([b["img"] for b in batch], dim=0),
        "cls": torch.cat(all_cls, dim=0) if all_cls else torch.zeros((0, 1), dtype=torch.float32),
        "bboxes": torch.cat(all_bboxes, dim=0) if all_bboxes else torch.zeros((0, 4), dtype=torch.float32),
        "batch_idx": torch.cat(all_batch_idx, dim=0) if all_batch_idx else torch.zeros(0, dtype=torch.float32),
        "masks": torch.stack(overlap_maps, dim=0),
        "sem_masks": torch.stack(sem_maps, dim=0),
        **_stack_geometry_fields(batch),
    }


def classification_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate classification adapter dicts into an Ultralytics-compatible batch.

    Stacks images into ``(B, C, H, W)`` and stacks class indices into a 1-D
    ``(B,)`` int64 tensor, matching what
    :class:`~ultralytics.models.yolo.classify.ClassificationValidator`
    expects in ``update_metrics``.
    """
    return _base_collate(
        batch,
        target_key="cls",
        collate_target_fn=lambda cls_list: torch.tensor(cls_list, dtype=torch.int64),
    )


def multilabel_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate multi-label classification adapter dicts into a batch.

    Stacks images into ``(B, C, H, W)`` and multi-hot labels into a float
    ``(B, num_classes)`` tensor.
    """
    return _base_collate(
        batch,
        target_key="cls",
        collate_target_fn=lambda cls_list: torch.stack(
            [torch.as_tensor(c, dtype=torch.float32) for c in cls_list], dim=0
        ),
    )


def semantic_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate semantic segmentation adapter dicts into a batch.

    Stacks images into ``(B, C, H, W)`` and dense semantic masks into a
    single ``(B, H, W)`` int32 tensor.
    """
    return _base_collate(
        batch,
        target_key="semantic_mask",
        collate_target_fn=lambda mask_list: torch.stack(
            [torch.as_tensor(m, dtype=torch.int32) for m in mask_list], dim=0
        ),
    )
