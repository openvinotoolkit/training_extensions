"""Evaluation methods for mmseg."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .mean_ap_seg import eval_segm
from .evaluator import Evaluator

__all__ = [
    "eval_segm",
    "Evaluator"
]
