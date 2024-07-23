# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Custom OTX Losses for Instance Segmentation."""

from .accuracy import accuracy
from .dice_loss import DiceLoss

__all__ = ["accuracy", "DiceLoss"]
