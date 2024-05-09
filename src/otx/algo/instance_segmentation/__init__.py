# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX instance segmentation models."""

from otx.core.model.utils.mmdet import DetDataPreprocessor

from . import mmdet

__all__ = ["mmdet", "DetDataPreprocessor"]
