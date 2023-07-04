"""Backbones for visual prompting model."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .vit import ViT, build_vit

__all__ = ["ViT", "build_vit"]
