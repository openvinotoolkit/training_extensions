"""Lightning-based Adapters."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

AVAILABLE = True
VERSION = None
DEBUG = None

try:
    import pytorch_lightning as pl

    VERSION = pl.__version__
    import os

    os.environ["FEATURE_FLAGS_OTX_VISUAL_PROMPTING_TASKS"] = "1"

    from .dataset import LightningDataset as Dataset
    from .engine import LightningEngine as Engine
    from .model import get_model, list_models

    __all__ = ["Dataset", "Engine", "get_model", "list_models"]

except ImportError as e:
    AVAILABLE = False
    DEBUG = e
