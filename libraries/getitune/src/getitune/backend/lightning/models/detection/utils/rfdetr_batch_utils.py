# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Batch object limiting utilities for RF-DETR and RF-DETR Instance Segmentation.

This module provides utilities to limit the number of objects in a training batch
to prevent OOM (Out of Memory) errors during RF-DETR training on dense datasets
like Visdrone (1000+ objects per image).

The limiting strategy prioritizes accuracy by:
1. First capping per-image objects to RF-DETR's query limit (300) - no accuracy loss
2. Reducing dense images while preserving sparse images
3. Using proportional reduction as a fallback

Note:
    This utility supports both detection and instance segmentation targets.
    When masks are present in targets, they are filtered along with boxes and labels.
    Masks should have shape (N, H, W) where N matches the number of boxes.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def limit_batch_objects(
    targets: list[dict[str, Any]],
    max_total: int,
    max_per_image: int = 300,
) -> list[dict[str, Any]]:
    """Limit batch objects with smart heuristic to prevent OOM.

    Strategy:
    1. First, cap any image exceeding max_per_image (300) - minimal accuracy loss
       since RF-DETR can only match 300 queries anyway.
    2. If still over budget, reduce dense images (>sparse_threshold) proportionally.
    3. Preserve sparse images (<sparse_threshold) as much as possible.

    Args:
        targets: List of target dicts with 'boxes', 'labels', and optionally 'masks'.
            Masks should have shape (N, H, W) where N matches the number of boxes.
        max_total: Maximum total objects allowed in the batch.
        max_per_image: Maximum objects per image (default: 300, RF-DETR query limit).

    Returns:
        Limited targets list with boxes, labels, and masks filtered consistently.

    Example:
        >>> # Detection targets
        >>> targets = [
        ...     {"boxes": torch.rand(500, 4), "labels": torch.randint(0, 10, (500,))},
        ...     {"boxes": torch.rand(100, 4), "labels": torch.randint(0, 10, (100,))},
        ... ]
        >>> limited = limit_batch_objects(targets, max_total=600, max_per_image=300)
        >>> sum(len(t["boxes"]) for t in limited) <= 600
        True
        >>>
        >>> # Instance segmentation targets (with masks)
        >>> targets = [
        ...     {"boxes": torch.rand(500, 4), "labels": torch.randint(0, 10, (500,)),
        ...      "masks": torch.rand(500, 64, 64) > 0.5},
        ... ]
        >>> limited = limit_batch_objects(targets, max_total=300, max_per_image=300)
        >>> len(limited[0]["masks"]) == len(limited[0]["boxes"])
        True
    """
    if not targets:
        return targets

    total_before = sum(len(t.get("boxes", [])) for t in targets)
    if total_before <= max_total:
        return targets  # No limiting needed

    # Step 1: Hard cap at max_per_image (minimal accuracy loss - RF-DETR can't use more)
    step1_targets = _cap_per_image(targets, max_per_image)
    total_after_step1 = sum(len(t.get("boxes", [])) for t in step1_targets)

    if total_after_step1 <= max_total:
        if total_after_step1 < total_before:
            logger.warning(
                "RF-DETR batch object limiting: capped %d -> %d objects (per-image cap at %d, max_total=%d)",
                total_before,
                total_after_step1,
                max_per_image,
                max_total,
            )
        return step1_targets

    # Step 2: Further reduce dense images proportionally, preserve sparse images
    sparse_threshold = max_per_image // 2  # 150 by default
    step2_targets = _reduce_dense_images(step1_targets, max_total, sparse_threshold)
    total_after_step2 = sum(len(t.get("boxes", [])) for t in step2_targets)

    if total_after_step2 <= max_total:
        logger.warning(
            "RF-DETR batch object limiting: capped %d -> %d objects "
            "(dense images reduced, sparse preserved, max_total=%d)",
            total_before,
            total_after_step2,
            max_total,
        )
        return step2_targets

    # Step 3: Fallback - proportionally limit all images
    final_targets = _proportional_limit(step2_targets, max_total)
    total_final = sum(len(t.get("boxes", [])) for t in final_targets)

    logger.warning(
        "RF-DETR batch object limiting: capped %d -> %d objects (proportional fallback, max_total=%d)",
        total_before,
        total_final,
        max_total,
    )
    return final_targets


def _cap_per_image(
    targets: list[dict[str, Any]],
    max_per_image: int,
) -> list[dict[str, Any]]:
    """Cap objects per image, keeping largest by area.

    Args:
        targets: List of target dicts.
        max_per_image: Maximum objects per image.

    Returns:
        Capped targets.
    """
    capped_targets = []
    for target in targets:
        boxes = target.get("boxes")
        if boxes is None or len(boxes) == 0:
            capped_targets.append(target)
            continue

        if len(boxes) > max_per_image:
            keep_indices = _get_largest_indices(boxes, max_per_image)
            capped_target = _subset_target(target, keep_indices, len(boxes))
            capped_targets.append(capped_target)
        else:
            capped_targets.append(target)

    return capped_targets


def _reduce_dense_images(
    targets: list[dict[str, Any]],
    max_total: int,
    sparse_threshold: int,
) -> list[dict[str, Any]]:
    """Reduce dense images while preserving sparse images.

    Args:
        targets: List of target dicts.
        max_total: Maximum total objects.
        sparse_threshold: Threshold below which an image is considered sparse.

    Returns:
        Reduced targets.
    """
    # Calculate sparse and dense totals
    sparse_count = sum(len(t.get("boxes", [])) for t in targets if len(t.get("boxes", [])) <= sparse_threshold)
    dense_budget = max_total - sparse_count

    if dense_budget <= 0:
        # Even sparse images exceed budget - caller should use proportional fallback
        return targets

    # Identify dense images
    dense_indices = [i for i, t in enumerate(targets) if len(t.get("boxes", [])) > sparse_threshold]
    dense_total = sum(len(targets[i].get("boxes", [])) for i in dense_indices)

    if dense_total <= dense_budget:
        return targets  # Dense images fit within budget

    # Proportionally reduce dense images
    reduced_targets = []
    for i, target in enumerate(targets):
        boxes = target.get("boxes")
        if boxes is None or len(boxes) == 0:
            reduced_targets.append(target)
            continue

        if i in dense_indices:
            # Calculate new count for this dense image
            new_count = max(sparse_threshold, int(len(boxes) * dense_budget / dense_total))
            new_count = min(new_count, len(boxes))

            if new_count < len(boxes):
                keep_indices = _get_largest_indices(boxes, new_count)
                reduced_target = _subset_target(target, keep_indices, len(boxes))
                reduced_targets.append(reduced_target)
            else:
                reduced_targets.append(target)
        else:
            reduced_targets.append(target)

    return reduced_targets


def _proportional_limit(
    targets: list[dict[str, Any]],
    max_total: int,
) -> list[dict[str, Any]]:
    """Fallback: proportionally limit all images.

    Args:
        targets: List of target dicts.
        max_total: Maximum total objects.

    Returns:
        Proportionally limited targets.
    """
    total = sum(len(t.get("boxes", [])) for t in targets)
    if total <= max_total:
        return targets

    ratio = max_total / total
    limited = []
    for target in targets:
        boxes = target.get("boxes")
        if boxes is None or len(boxes) == 0:
            limited.append(target)
            continue

        new_count = max(1, int(len(boxes) * ratio))
        if new_count < len(boxes):
            keep_indices = _get_largest_indices(boxes, new_count)
            limited_target = _subset_target(target, keep_indices, len(boxes))
            limited.append(limited_target)
        else:
            limited.append(target)

    return limited


def _get_largest_indices(boxes: torch.Tensor, k: int) -> torch.Tensor:
    """Get indices of k largest boxes by area.

    Args:
        boxes: Boxes tensor in cxcywh format (N, 4).
        k: Number of boxes to keep.

    Returns:
        Sorted indices of k largest boxes.
    """
    areas = boxes[:, 2] * boxes[:, 3]  # w * h
    return torch.topk(areas, k=k, largest=True)[1].sort()[0]


def _subset_target(
    target: dict[str, Any],
    indices: torch.Tensor,
    original_length: int,
) -> dict[str, Any]:
    """Create a subset of target dict using given indices.

    This function handles all tensor fields that have the same first dimension
    as the number of objects, including boxes, labels, and masks.

    Args:
        target: Target dict with 'boxes', 'labels', and optionally 'masks', etc.
        indices: Indices to keep.
        original_length: Original number of boxes/objects.

    Returns:
        Subset target dict with all relevant tensors filtered.
    """
    subset = {}
    for key, val in target.items():
        if isinstance(val, torch.Tensor) and val.ndim >= 1 and val.shape[0] == original_length:
            subset[key] = val[indices]
        else:
            subset[key] = val
    return subset
