"""Adapter of Visual-Prompt."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

AVAILABLE = True
VERSION = None
DEBUG = None

try:
    import os
    os.environ["FEATURE_FLAGS_OTX_VISUAL_PROMPTING_TASKS"] = "1"

    from .dataset import VisualPromptDataset as Dataset
    from .engine import VisualPromptEngine as Engine
    from .model import get_model, list_models

    __all__ = ["Dataset", "Engine", "get_model", "list_models"]
    VERSION = "0.0.1"

except ImportError as e:
    AVAILABLE = False
    DEBUG = e
