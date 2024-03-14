# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Custom Losses for OTX segmentation model."""

from .cross_entropy_loss_with_ignore import CrossEntropyLossWithIgnore

__all__ = ["CrossEntropyLossWithIgnore"]
