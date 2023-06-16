"""Initial file for mmdeploy ops."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .custom_ops import grid_sampler__default, squeeze__default

__all__ = ["squeeze__default", "grid_sampler__default"]
