"""Encoders for visual prompting model."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .sam_image_encoder import SAMImageEncoderViT
from .sam_prompt_encoder import SAMPromptEncoder

__all__ = ["SAMImageEncoderViT", "SAMPromptEncoder"]
