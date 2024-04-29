# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX instance segmentation models."""

from otx.core.model.utils.mmdet import (
    DetDataPreprocessor,  # TODO(Eugene): Remove this after decoupling det data preprocessor
)

from . import heads, mmdet

__all__ = ["heads", "mmdet", "DetDataPreprocessor"]
