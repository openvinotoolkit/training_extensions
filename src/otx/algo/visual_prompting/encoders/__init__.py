# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Encoder modules for OTX visual prompting model."""

from .sam_image_encoder import SAMImageEncoder
from .sam_prompt_encoder import SAMPromptEncoder

__all__ = ["SAMImageEncoder", "SAMPromptEncoder"]
