"""Backbones for Lightning Custom Model."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .tiny_vit import build_tiny_vit
from .vit import build_vit

__all__ = ["build_tiny_vit", "build_vit"]
