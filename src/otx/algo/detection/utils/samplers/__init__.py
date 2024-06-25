# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Samplers for detection task."""

from .base_sampler import PseudoSampler, RandomSampler

__all__ = ["PseudoSampler", "RandomSampler"]
