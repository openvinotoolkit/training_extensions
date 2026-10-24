# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""EdgeCrafter criterion implementation.

Modified from EdgeCrafter (https://github.com/Intellindust-AI-Lab/EdgeCrafter).
Modified from D-FINE (https://github.com/Peterande/D-FINE).
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from getitune.backend.lightning.models.common.utils.assigners.hungarian_matcher import HungarianMatcher

from .deim_loss import DEIMCriterion


class ECCriterion(DEIMCriterion):
    """EdgeCrafter criterion extending DEIMCriterion with instance-segmentation mask losses.

    Adds sigmoid-BCE and Dice losses for matched prediction/GT mask pairs on top of the
    VFL, MAL, box regression, FGL, and DDF losses inherited from :class:`DEIMCriterion`.

    Mask losses are computed only when ``pred_masks`` is present in ``outputs``,
    so auxiliary encoder branches (which do not produce masks) are handled
    transparently without extra bookkeeping.

    Args:
        weight_dict: Loss weight dictionary; may include ``loss_mask_ce`` and
            ``loss_mask_dice`` keys for mask losses.
        alpha: VFL/MAL alpha parameter. Defaults to 0.2.
        gamma: VFL/MAL gamma parameter. Defaults to 2.0.
        num_classes: Number of object classes. Defaults to 80.
        reg_max: Bin count for distribution-based box regression. Defaults to 32.
        matcher_cost_dict: Optional custom cost dictionary for the Hungarian matcher.
            When provided, overrides the default detection matcher.
    """

    def __init__(
        self,
        weight_dict: dict[str, int | float],
        alpha: float = 0.2,
        gamma: float = 2.0,
        num_classes: int = 80,
        reg_max: int = 32,
        matcher_cost_dict: dict[str, int | float] | None = None,
    ) -> None:
        super().__init__(weight_dict, alpha, gamma, num_classes, reg_max)
        if matcher_cost_dict is not None:
            self.matcher = HungarianMatcher(cost_dict=matcher_cost_dict)

    def loss_masks(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
        num_boxes: int,
    ) -> dict[str, Tensor]:
        """Sigmoid-BCE and Dice losses on matched prediction/GT mask pairs.

        Silently returns an empty dict when ``pred_masks`` is absent in
        ``outputs`` (e.g. for encoder auxiliary branches or pure detection).

        Args:
            outputs: Model output dict; must contain ``pred_masks`` [B, Q, H, W]
                when mask supervision is desired.
            targets: Per-image target dicts; must contain ``masks`` [N, Ht, Wt].
            indices: Hungarian-matched (src_idx, tgt_idx) pairs per image.
            num_boxes: Normalisation denominator (total matched boxes).

        Returns:
            Dict with ``loss_mask_ce`` and ``loss_mask_dice``, or empty dict.
        """
        pred_masks = outputs.get("pred_masks")
        if pred_masks is None:
            return {}

        idx = self._get_src_permutation_idx(indices)
        src_masks = pred_masks[idx]  # [N, H_m, W_m]

        if src_masks.numel() == 0:
            return {
                "loss_mask_ce": src_masks.sum(),
                "loss_mask_dice": src_masks.sum(),
            }

        # Gather matched GT masks and resize to prediction spatial size
        target_masks = torch.cat(
            [t["masks"][j] for t, (_, j) in zip(targets, indices)],
            dim=0,
        ).float()  # [N, Ht, Wt]

        h_m, w_m = src_masks.shape[-2:]
        if target_masks.shape[-2:] != (h_m, w_m):
            target_masks = F.interpolate(
                target_masks.unsqueeze(1),
                size=(h_m, w_m),
                mode="nearest",
            ).squeeze(1)

        # Sigmoid BCE loss
        loss_ce = F.binary_cross_entropy_with_logits(src_masks, target_masks, reduction="none")
        loss_ce = loss_ce.flatten(1).mean(1).sum() / num_boxes

        # Dice loss
        pred_sig = src_masks.sigmoid().flatten(1)
        tgt_flat = target_masks.flatten(1)
        numerator = 2 * (pred_sig * tgt_flat).sum(-1)
        denominator = pred_sig.sum(-1) + tgt_flat.sum(-1)
        loss_dice = (1 - (numerator + 1) / (denominator + 1)).sum() / num_boxes

        return {"loss_mask_ce": loss_ce, "loss_mask_dice": loss_dice}

    @property
    def _available_losses(self) -> tuple[Callable]:  # type: ignore[return-value]
        return (  # pyrefly: ignore[bad-return]
            self.loss_boxes,
            self.loss_labels_mal,
            self.loss_local,
            self.loss_masks,
        )
