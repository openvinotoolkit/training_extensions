# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for mask target computation robustness."""

from __future__ import annotations

import numpy as np
import torch
from torchvision import tv_tensors

from getitune.data.utils.structures.mask.mask_target import mask_target_single


def _proposals(n: int) -> torch.Tensor:
    """Return ``n`` proposals covering the whole 32x32 image."""
    return torch.tensor([[0.0, 0.0, 32.0, 32.0]]).repeat(n, 1)


class TestMaskTargetSingle:
    """Tests for mask_target_single."""

    def test_plain_tensor_masks_produce_correct_count(self) -> None:
        """A plain ``torch.Tensor`` gt_masks must yield one target per proposal.

        Slicing a ``tv_tensors.Mask`` returns a plain ``torch.Tensor`` (losing the
        subclass). Previously mask_target_single returned *zero* targets for such
        inputs, desynchronising the mask prediction/target counts and crashing the
        mask BCE loss with "Target size ... must be the same as input size ...".
        A plain tensor of shape (N, H, W) is a valid mask stack and must be used.
        """
        gt_masks = torch.zeros((2, 32, 32), dtype=torch.uint8)
        gt_masks[0, 4:20, 4:20] = 1
        gt_masks[1, 10:30, 10:30] = 1
        assert not isinstance(gt_masks, tv_tensors.Mask)  # sanity: plain tensor

        pos_proposals = _proposals(3)
        pos_assigned_gt_inds = torch.tensor([0, 1, 0])

        targets = mask_target_single(
            pos_proposals,
            pos_assigned_gt_inds,
            gt_masks,  # type: ignore[arg-type]
            mask_size=[28, 28],
            meta_info={"img_shape": (32, 32)},
        )

        # One target per positive proposal (not zero!), at the requested mask size.
        assert targets.shape == (3, 28, 28)

    def test_tv_tensor_masks_still_work(self) -> None:
        """The normal ``tv_tensors.Mask`` path is unaffected."""
        gt_masks = tv_tensors.Mask(torch.zeros((2, 32, 32), dtype=torch.uint8))
        gt_masks[0, 4:20, 4:20] = 1
        gt_masks[1, 10:30, 10:30] = 1

        targets = mask_target_single(
            _proposals(2),
            torch.tensor([0, 1]),
            gt_masks,
            mask_size=[28, 28],
            meta_info={"img_shape": (32, 32)},
        )
        assert targets.shape == (2, 28, 28)

    def test_empty_masks_return_zero_targets(self) -> None:
        """An empty mask stack still returns an empty target tensor."""
        gt_masks = tv_tensors.Mask(torch.zeros((0, 32, 32), dtype=torch.uint8))
        targets = mask_target_single(
            _proposals(0),
            torch.zeros((0,), dtype=torch.long),
            gt_masks,
            mask_size=[28, 28],
            meta_info={"img_shape": (32, 32)},
        )
        assert targets.shape[0] == 0

    def test_empty_masks_with_positive_proposals_keep_count(self) -> None:
        """Empty gt_masks but N positive proposals must yield N (empty) targets.

        ``mask_preds`` is produced for every positive proposal, so returning zero
        targets here would desync the counts and crash the mask loss with
        "Target size ... must be the same as input size ...".
        """
        gt_masks = tv_tensors.Mask(torch.zeros((0, 32, 32), dtype=torch.uint8))
        targets = mask_target_single(
            _proposals(2),
            torch.zeros((2,), dtype=torch.long),
            gt_masks,
            mask_size=[28, 28],
            meta_info={"img_shape": (32, 32)},
        )
        # One (empty) target per positive proposal, not zero.
        assert targets.shape == (2, 28, 28)

    def test_unsupported_mask_type_keeps_count(self) -> None:
        """A genuinely unsupported gt_masks type must still return N targets.

        Even when the mask type cannot be interpreted, the target count must match
        the number of positive proposals so the mask loss does not crash.
        """
        gt_masks = np.zeros((2, 32, 32), dtype=np.uint8)  # numpy: unsupported type

        targets = mask_target_single(
            _proposals(3),
            torch.tensor([0, 1, 0]),
            gt_masks,  # type: ignore[arg-type]
            mask_size=[28, 28],
            meta_info={"img_shape": (32, 32)},
        )
        # One (empty) target per positive proposal, keeping counts in sync.
        assert targets.shape == (3, 28, 28)
