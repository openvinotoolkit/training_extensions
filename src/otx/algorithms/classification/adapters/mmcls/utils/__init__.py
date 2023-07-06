"""OTX Adapters - mmdet.utils."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .builder import build_classifier
from .config_utils import patch_datasets, patch_evaluation

__all__ = [
    "build_classifier",
    "patch_datasets",
    "patch_evaluation",
]
