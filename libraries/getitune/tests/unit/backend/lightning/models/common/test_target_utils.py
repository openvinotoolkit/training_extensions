# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared annotation-alignment helper."""

from __future__ import annotations

import torch
from torchvision import tv_tensors

from getitune.backend.lightning.models.common.target_utils import align_sample_batch_annotations
from getitune.data.entity.base import ImageInfo
from getitune.data.entity.sample import SampleBatch


def _bbox(n: int) -> tv_tensors.BoundingBoxes:
    data = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32).repeat(n, 1) if n else torch.zeros((0, 4))
    return tv_tensors.BoundingBoxes(  # pyrefly: ignore[no-matching-overload]
        data, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(64, 64)
    )


def _mask(n: int) -> tv_tensors.Mask:
    return tv_tensors.Mask(torch.zeros((n, 64, 64), dtype=torch.uint8))


def _masks_with_boxes(boxes: list[tuple[int, int, int, int]]) -> tv_tensors.Mask:
    """Build an ``(N, 64, 64)`` mask stack where mask i is a filled rectangle."""
    data = torch.zeros((len(boxes), 64, 64), dtype=torch.uint8)
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        data[idx, y1:y2, x1:x2] = 1
    return tv_tensors.Mask(data)


def _batch(bbox_counts, label_counts, mask_counts=None) -> SampleBatch:
    bs = len(bbox_counts)
    return SampleBatch(
        images=torch.zeros((bs, 3, 64, 64)),
        bboxes=[_bbox(c) for c in bbox_counts],
        labels=[torch.zeros(c, dtype=torch.long) for c in label_counts],
        masks=None if mask_counts is None else [_mask(c) for c in mask_counts],
        imgs_info=[
            ImageInfo(img_idx=i, img_shape=(64, 64), ori_shape=(64, 64))  # pyrefly: ignore[no-matching-overload]
            for i in range(bs)
        ],
    )


class TestAlignSampleBatchAnnotations:
    """Tests for align_sample_batch_annotations."""

    def test_consistent_counts_are_untouched(self) -> None:
        """When all counts already match, the annotations are left as-is."""
        batch = _batch(bbox_counts=[2, 1], label_counts=[2, 1], mask_counts=[2, 1])
        align_sample_batch_annotations(batch)
        assert batch.bboxes is not None
        assert batch.masks is not None
        assert batch.labels is not None
        assert [b.shape[0] for b in batch.bboxes] == [2, 1]
        assert [m.shape[0] for m in batch.masks] == [2, 1]
        assert [labels.shape[0] for labels in batch.labels] == [2, 1]

    def test_detection_boxes_labels_mismatch_is_aligned(self) -> None:
        """A boxes-vs-labels divergence is trimmed to the per-image minimum."""
        batch = _batch(bbox_counts=[3, 1], label_counts=[1, 1])
        align_sample_batch_annotations(batch)
        assert batch.bboxes is not None
        assert batch.labels is not None
        assert batch.bboxes[0].shape[0] == 1
        assert batch.labels[0].shape[0] == 1
        # second image already consistent
        assert batch.bboxes[1].shape[0] == 1

    def test_instance_seg_masks_mismatch_is_aligned(self) -> None:
        """A masks-vs-boxes divergence is aligned across boxes/labels/masks."""
        # Image 0: 1 box / 1 label but 3 masks; Image 1: 2 boxes/labels but 1 mask.
        batch = _batch(bbox_counts=[1, 2], label_counts=[1, 2], mask_counts=[3, 1])
        align_sample_batch_annotations(batch)
        assert batch.bboxes is not None
        assert batch.labels is not None
        assert batch.masks is not None
        for i in range(2):
            n = batch.bboxes[i].shape[0]
            assert batch.labels[i].shape[0] == n
            assert batch.masks[i].shape[0] == n
        assert batch.bboxes[0].shape[0] == 1
        assert batch.bboxes[1].shape[0] == 1

    def test_no_annotations_is_noop(self) -> None:
        """Batches without instance annotations are handled gracefully."""
        batch = SampleBatch(images=torch.zeros((1, 3, 64, 64)))
        # Should not raise.
        assert align_sample_batch_annotations(batch) is batch

    def test_trimmed_masks_preserve_tv_tensor_type(self) -> None:
        """Trimming a mismatched sample must keep masks as ``tv_tensors.Mask``.

        Slicing a ``tv_tensors.Mask`` returns a *plain* ``torch.Tensor``. If the
        subclass is lost, the Mask R-CNN mask head (``mask_target_single``)
        silently drops that image's mask targets and desynchronises the
        prediction/target counts, crashing the mask BCE loss with
        "Target size ... must be the same as input size ...". The alignment
        helper must therefore re-wrap the trimmed masks.
        """
        batch = _batch(bbox_counts=[2], label_counts=[2], mask_counts=[3])
        align_sample_batch_annotations(batch)
        assert batch.masks is not None
        assert isinstance(batch.masks[0], tv_tensors.Mask)
        assert batch.masks[0].shape[0] == 2
        assert batch.bboxes is not None
        assert isinstance(batch.bboxes[0], tv_tensors.BoundingBoxes)

    def test_trimmed_detection_boxes_preserve_tv_tensor_type(self) -> None:
        """Detection box trimming must also preserve ``tv_tensors.BoundingBoxes``."""
        batch = _batch(bbox_counts=[3], label_counts=[1])
        align_sample_batch_annotations(batch)
        assert batch.bboxes is not None
        assert isinstance(batch.bboxes[0], tv_tensors.BoundingBoxes)
        assert batch.bboxes[0].canvas_size == (64, 64)
        assert batch.bboxes[0].shape[0] == 1

    def test_instance_seg_boxes_are_recomputed_from_masks(self) -> None:
        """On divergence, inst-seg boxes are rebuilt from masks (correct correspondence).

        This is the key guarantee for the Mask R-CNN mask head: after alignment
        ``box[i]`` must be the tight bounding box of ``mask[i]`` so a proposal
        matched to a GT box gathers the *matching* GT mask. Blindly trimming
        boxes to the first ``n`` would pair a box with the wrong mask when a
        middle instance is dropped from only one field.
        """
        mask_boxes = [(2, 3, 12, 20), (30, 5, 45, 40), (50, 50, 60, 62)]
        masks = _masks_with_boxes(mask_boxes)
        # Stored boxes deliberately unrelated to the mask rectangles so a plain
        # trim would leave boxes that do not match the surviving masks.
        stored = tv_tensors.BoundingBoxes(  # pyrefly: ignore[no-matching-overload]
            torch.tensor([[0, 0, 1, 1], [0, 0, 2, 2]], dtype=torch.float32),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(64, 64),
        )
        batch = SampleBatch(
            images=torch.zeros((1, 3, 64, 64)),
            bboxes=[stored],
            labels=[torch.zeros(2, dtype=torch.long)],
            masks=[masks],
            imgs_info=[
                ImageInfo(img_idx=0, img_shape=(64, 64), ori_shape=(64, 64))  # pyrefly: ignore[no-matching-overload]
            ],
        )

        align_sample_batch_annotations(batch)

        assert batch.bboxes is not None
        assert batch.masks is not None
        assert batch.labels is not None
        # Everything is trimmed to the common minimum (2).
        assert batch.bboxes[0].shape[0] == 2
        assert batch.masks[0].shape[0] == 2
        assert batch.labels[0].shape[0] == 2
        # Boxes now match the (trimmed) masks' tight bounding boxes exactly.
        for i in range(2):
            x1, y1, x2, y2 = mask_boxes[i]
            # masks_to_boxes uses inclusive max pixel indices (x2 - 1, y2 - 1).
            assert batch.bboxes[0][i].tolist() == [float(x1), float(y1), float(x2 - 1), float(y2 - 1)]
        assert isinstance(batch.bboxes[0], tv_tensors.BoundingBoxes)
        assert batch.bboxes[0].canvas_size == (64, 64)
