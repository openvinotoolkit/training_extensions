"""Init file for otx.algorithms.common.adapters.mmdeploy.utils."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .mmdeploy import is_mmdeploy_enabled
from .utils import numpy_2_list, sync_batchnorm_2_batchnorm

__all__ = [
    "is_mmdeploy_enabled",
    "sync_batchnorm_2_batchnorm",
    "numpy_2_list",
]
