# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for base NN segmentation models."""

from .base_model import BaseSegmModel
from .mean_teacher import MeanTeacher

__all__ = ["BaseSegmModel", "MeanTeacher"]
