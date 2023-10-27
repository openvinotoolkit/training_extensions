"""Visual prompting model."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from otx.v2.adapters.torch.lightning.visual_prompt.registry import VisualPromptRegistry

from .visual_prompters import SegmentAnything

# NOTE: Register the model with the Registry to make it available via the config API.
MODELS = VisualPromptRegistry()
MODELS.register_module(name="SAM", module=SegmentAnything)
