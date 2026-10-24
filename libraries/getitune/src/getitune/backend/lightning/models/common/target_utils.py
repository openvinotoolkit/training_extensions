# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for preparing model training targets.

This module provides defensive utilities used by the detection and instance
segmentation models when converting a :class:`~getitune.data.entity.sample.SampleBatch`
into model-specific training targets.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from getitune.data.entity.sample import SampleBatch

logger = logging.getLogger(__name__)


def align_sample_batch_annotations(entity: SampleBatch) -> SampleBatch:
    """Align per-image annotation counts (bboxes/labels/masks/keypoints) in place.

    Many detection and instance-segmentation criteria concatenate a sample's
    per-image ``boxes``, ``labels`` and ``masks`` and require their counts to
    match:

    * DETR-style models (RF-DETR, RT-DETR, DEIM/D-FINE, DEIMv2) feed them to a
      Hungarian matcher that adds per-target cost matrices element-wise. A count
      mismatch raises ``RuntimeError: The size of tensor a (N) must match the
      size of tensor b (M) ...`` deep inside the matcher.
    * mmdet-based models (Mask R-CNN, RTMDet-Inst, ATSS, SSD, YOLOX) build
      ``InstanceData`` which enforces that all instance-level fields share the
      same length. For mask heads, positive proposals are matched to ground
      truth by *box* IoU and the resulting index set is used to gather *masks*
      (``gt_masks[pos_assigned_gt_inds]``); if a sample has more boxes than
      masks this indexes out of range and triggers a CUDA device-side assert.

    The data pipeline should already guarantee consistent counts, but geometric
    transforms can drop one annotation type without dropping the paired ones.
    In particular, tiling filters bounding boxes (``BboxTiler``) and instance
    masks with *independent* criteria — boxes by their stored (e.g. COCO)
    coordinates and masks by their rasterised pixels — so an instance straddling
    a tile boundary can be kept for one field and dropped for the other, leaving
    ``len(boxes) != len(masks)``.

    When counts diverge this helper realigns the affected image and logs a
    warning so training degrades gracefully instead of crashing inside the loss.

    Alignment strategy (only applied to images whose counts actually diverge —
    the common consistent path is left completely untouched):

    * **Instance segmentation** (masks present): the masks are treated as the
      source of truth. Every field is trimmed to the common minimum count and
      the boxes are then *recomputed from the (trimmed) masks*. This guarantees
      ``box[i]`` is exactly the tight bounding box of ``mask[i]``, preserving the
      box→mask correspondence the Mask R-CNN mask head relies on. Blindly
      trimming boxes to the first ``n`` entries would instead risk pairing
      ``box[i]`` with a *different* instance's mask when a middle instance was
      dropped from only one of the two fields.
    * **Detection / other** (no masks): each field is trimmed to the common
      minimum count (boxes and labels are produced in the same instance order,
      so trimming trailing extras keeps them aligned).

    Args:
        entity: The batch to align. Modified in place.

    Returns:
        The same ``entity`` instance (for convenient chaining).
    """
    bboxes = entity.bboxes
    labels = entity.labels
    masks = entity.masks
    keypoints = entity.keypoints

    # Nothing to align if there are no instance-level annotations.
    if bboxes is None and labels is None and masks is None and keypoints is None:
        return entity

    for i in range(entity.batch_size):
        counts: list[int] = [
            int(field[i].shape[0])
            for field in (bboxes, labels, masks, keypoints)
            if field is not None and i < len(field) and field[i] is not None
        ]

        if not counts:
            continue

        n = min(counts)
        if all(c == n for c in counts):
            continue  # already consistent — leave untouched

        logger.warning(
            "Sample %d has mismatched annotation counts "
            "(bboxes=%s, labels=%s, masks=%s, keypoints=%s); aligning to %d.",
            i,
            None if bboxes is None else int(bboxes[i].shape[0]),
            None if labels is None else int(labels[i].shape[0]),
            None if masks is None else int(masks[i].shape[0]),
            None if keypoints is None else int(keypoints[i].shape[0]),
            n,
        )

        masks_present = masks is not None and i < len(masks) and masks[i] is not None

        if labels is not None and labels[i] is not None:
            labels[i] = _trim_preserving_type(labels[i], n)
        if keypoints is not None and keypoints[i] is not None:
            keypoints[i] = _trim_preserving_type(keypoints[i], n)
        if masks_present:
            masks[i] = _trim_preserving_type(masks[i], n)  # type: ignore[index]  # pyrefly: ignore[unsupported-operation]

        if bboxes is not None and bboxes[i] is not None:
            if masks_present:
                # Instance segmentation: rebuild boxes from the trimmed masks so
                # box[i] is guaranteed to correspond to mask[i].
                bboxes[i] = _boxes_from_masks(masks[i], bboxes[i])  # type: ignore[index]
            else:
                bboxes[i] = _trim_preserving_type(bboxes[i], n)  # pyrefly: ignore[unsupported-operation]

    return entity


def _trim_preserving_type(field_value: Tensor, n: int) -> Tensor:
    """Trim ``field_value`` to its first ``n`` rows, preserving any tv_tensor subclass.

    Slicing a ``torchvision.tv_tensors`` instance (e.g. ``tv_tensors.Mask`` or
    ``tv_tensors.BoundingBoxes``) returns a *plain* ``torch.Tensor`` — the
    subclass and its metadata (``format``/``canvas_size`` for boxes) are lost.
    Downstream consumers rely on the subclass: in particular the Mask R-CNN mask
    head silently drops a whole image's mask targets when ``gt_masks`` is not a
    ``tv_tensors.Mask`` (see ``mask_target_single``), which corrupts the
    prediction/target counts and crashes the mask loss. Re-wrapping the sliced
    tensor with :func:`torchvision.tv_tensors.wrap` keeps the type intact.
    """
    trimmed = field_value[:n]
    from torchvision import tv_tensors

    if isinstance(field_value, tv_tensors.TVTensor) and not isinstance(trimmed, type(field_value)):
        # pyrefly mis-resolves ``tv_tensors.wrap`` to ``torch._dynamo`` due to a
        # name collision and rejects the valid ``like=`` kwarg; the call is correct.
        trimmed = tv_tensors.wrap(trimmed, like=field_value)  # pyrefly: ignore[bad-argument-type, unexpected-keyword]
    return trimmed  # pyrefly: ignore[bad-return]


def _boxes_from_masks(masks: Tensor, reference_bboxes: Tensor) -> Tensor:
    """Return tight bounding boxes for ``masks`` as ``tv_tensors.BoundingBoxes``.

    The result mirrors the format/canvas of ``reference_bboxes`` so it is a
    drop-in replacement inside a :class:`~getitune.data.entity.sample.SampleBatch`.

    Args:
        masks: Instance masks of shape ``(N, H, W)``.
        reference_bboxes: The existing (possibly mis-counted) boxes, used only to
            copy ``format``/``canvas_size``/``dtype``/``device`` metadata.

    Returns:
        ``tv_tensors.BoundingBoxes`` of shape ``(N, 4)`` in XYXY format.
    """
    # Local imports keep this module import-light and avoid any import cycle
    # between the model-common package and the data package.
    from torchvision import tv_tensors

    from getitune.data.utils.structures.mask.mask_target import masks_to_boxes

    canvas_size = getattr(reference_bboxes, "canvas_size", tuple(masks.shape[-2:]))
    xyxy = masks_to_boxes(masks, dtype=torch.float32).to(masks.device)
    return tv_tensors.BoundingBoxes(
        xyxy,
        format=tv_tensors.BoundingBoxFormat.XYXY,
        canvas_size=canvas_size,  # type: ignore[arg-type]
        dtype=torch.float32,
    )
