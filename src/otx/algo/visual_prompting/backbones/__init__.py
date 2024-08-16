# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Backbone modules for OTX visual prompting model."""

from .tiny_vit import TinyViT
from .vit import ViT

__all__ = ["TinyViT", "ViT"]
