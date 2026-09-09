# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Loss criteria for Ultralytics-backed classification models."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional


class MultiLabelClassificationLoss(nn.Module):
    """Binary cross-entropy loss for multi-label classification.

    Mirrors the upstream ``v8ClassificationLoss`` call signature so it can be
    dropped in as ``model.criterion`` for YOLO classification models.
    """

    def forward(
        self,
        preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute BCE loss between raw logits and multi-hot targets.

        Args:
            preds: Raw classification logits, or a list/tuple containing them.
            batch: Ultralytics batch dict with ``"cls"`` multi-hot targets.

        Returns:
            Tuple of (loss, dict of detached per-component losses).
        """
        logits = preds[0] if isinstance(preds, (list, tuple)) else preds
        targets = batch["cls"].float()
        loss = functional.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
        return loss, {"loss": loss.detach()}
