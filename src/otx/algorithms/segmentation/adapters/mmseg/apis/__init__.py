"""Adapters of classification - mmseg."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .simple_train_xpu import train_segmentor_debug
from .train import train_segmentor

__all__ = ["train_segmentor", "train_segmentor_debug"]
