"""Configuration file of OTX Visual Prompting."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from attr import attrs

from otx.algorithms.visual_prompting.configs.base import VisualPromptingBaseConfig


@attrs
class VisualPromptingConfig(VisualPromptingBaseConfig):
    """Configurable parameters for Visual Prompting task."""
