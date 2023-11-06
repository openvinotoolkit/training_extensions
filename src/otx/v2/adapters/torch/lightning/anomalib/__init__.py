"""Adapter of Anomalib."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

AVAILABLE = True
VERSION = None
DEBUG = None

try:
    import anomalib

    VERSION = anomalib.__version__

    from .dataset import AnomalibDataset as Dataset
    from .engine import AnomalibEngine as Engine
    from .model import get_model, list_models

    __all__ = ["Dataset", "Engine", "get_model", "list_models"]

except ImportError as e:
    AVAILABLE = False
    DEBUG = e
