# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Custom Losses for OTX segmentation model."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .cross_entropy_loss_with_ignore import CrossEntropyLossWithIgnore

if TYPE_CHECKING:
    from torch import nn

__all__ = ["CrossEntropyLossWithIgnore", "create_criterion"]


def create_criterion(losses: list | str, params: dict | None = None) -> nn.Module:
    """Create loss function by name."""
    if isinstance(losses, list):
        creterions: list = []
        for loss in losses:
            loss_type = loss["type"]
            params = loss.get("params", {})
            creterions.append(create_criterion(loss_type, params))
        return creterions

    if isinstance(losses, str):
        params = {} if params is None else params
        if losses == "CrossEntropyLoss":
            return CrossEntropyLossWithIgnore(**params)

        msg = f"Unknown loss type: {losses}"
        raise ValueError(msg)

    msg = "losses should be a dict or a string"
    raise ValueError(msg)
