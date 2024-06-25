# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Custom backbone implementations."""

from .resnet import ResNet
from .resnext import ResNeXt

__all__ = ["ResNet", "ResNeXt"]
