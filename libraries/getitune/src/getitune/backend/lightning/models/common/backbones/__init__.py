# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom backbone implementations."""

from .pytorchcv_backbones import build_model_including_pytorchcv
from .resnet import ResNet
from .resnext import ResNeXt

__all__ = ["ResNeXt", "ResNet", "build_model_including_pytorchcv"]
