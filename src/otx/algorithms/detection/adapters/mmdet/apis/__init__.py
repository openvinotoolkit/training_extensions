"""Adapters of classification - mmdet."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .train import train_detector
from .simple_train_xpu import train_detector_debug

__all__ = ["train_detector", "train_detector_debug"]
