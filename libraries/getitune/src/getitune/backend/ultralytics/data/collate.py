# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom collate function producing Ultralytics batch dicts."""

from __future__ import annotations

from typing import Any, Callable

import torch


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


def instance_seg_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate instance-segmentation adapter dicts into an Ultralytics-compatible batch.

    Stacks images into ``(B, C, H, W)`` and concatenates per-sample
    annotations with correct per-image ``batch_idx``.

    Instance masks are collated in **overlap format** (matching upstream
    Ultralytics ``overlap_mask=True``): a single ``(B, H, W)`` index map
    per batch where pixel values 1..N identify instance ownership.
    Instances are sorted by area descending so smaller masks overwrite
    larger ones in overlapping regions.
    """
    all_cls, all_bboxes, all_batch_idx = _collect_detection_targets(batch)

    overlap_maps: list[torch.Tensor] = []
    sem_maps: list[torch.Tensor] = []

    for i, b in enumerate(batch):
        masks = b["masks"]
        if not isinstance(masks, torch.Tensor):
            masks = torch.as_tensor(masks, dtype=torch.float32)

        n_inst = masks.shape[0]
        if n_inst == 0:
            if masks.ndim == 3:
                h, w = masks.shape[1], masks.shape[2]
            else:
                h, w = 1, 1
            overlap_maps.append(torch.zeros((h, w), dtype=torch.uint8))
            sem_maps.append(torch.zeros((h, w), dtype=torch.float32))
            continue

        # Sort by area descending (larger first, smaller overwrite).
        areas = masks.float().sum(dim=(1, 2))  # (N,)
        sorted_idx = torch.argsort(areas, descending=True)
        masks = masks[sorted_idx]

        # Reorder cls/bboxes for this image to match sorted order.
        cls_i = all_cls[i][sorted_idx]  # (N, 1)
        bboxes_i = all_bboxes[i][sorted_idx]  # (N, 4)
        # Update the lists so that the concatenated batch reflects sorted order.
        all_cls[i] = cls_i
        all_bboxes[i] = bboxes_i

        # Build overlap index map: paint instances 1..N (smaller overwrite larger).
        h, w = masks.shape[1], masks.shape[2]
        index_map = torch.zeros((h, w), dtype=torch.uint8)
        for k in range(n_inst):
            index_map[masks[k] > 0] = k + 1
        overlap_maps.append(index_map)

        # Build sem_masks from index map + class labels.
        cls_flat = cls_i.squeeze(-1)  # (N,)
        sem = torch.zeros((h, w), dtype=torch.float32)
        fg = index_map > 0
        if fg.any():
            sem[fg] = cls_flat[(index_map[fg].long() - 1)]
        sem_maps.append(sem)

    return {
        "img": torch.stack([b["img"] for b in batch], dim=0),
        # Rebuilt from the mask-area-sorted per-image lists.
        "cls": torch.cat(all_cls, dim=0) if all_cls else torch.zeros((0, 1), dtype=torch.float32),
        "bboxes": torch.cat(all_bboxes, dim=0) if all_bboxes else torch.zeros((0, 4), dtype=torch.float32),
        "batch_idx": torch.cat(all_batch_idx, dim=0) if all_batch_idx else torch.zeros(0, dtype=torch.float32),
        "masks": torch.stack(overlap_maps, dim=0),  # (B, H, W)
        "sem_masks": torch.stack(sem_maps, dim=0),  # (B, H, W)
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
