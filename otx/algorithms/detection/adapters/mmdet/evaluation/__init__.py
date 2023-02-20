"""Evaluation methods for mmseg."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .mean_ap_seg import eval_segm

__all__ = [
    "eval_segm",
]
