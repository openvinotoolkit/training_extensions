# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Custom Losses for OTX segmentation model."""

from .cross_entropy_loss_with_ignore import CrossEntropyLossWithIgnore

__all__ = ["CrossEntropyLossWithIgnore", "create_criterion"]


def create_criterion(type: str, **kwargs):
    from torchvision.ops import focal_loss
    """Create loss function by name."""
    if type == "CrossEntropyLoss":
        return CrossEntropyLossWithIgnore(**kwargs)
    else:
        raise NotImplementedError
