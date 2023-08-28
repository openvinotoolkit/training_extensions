"""Adapter of Anomalib."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

AVAILABLE = True
VERSION = None

try:
    import anomalib  # noqa: F401

    VERSION = anomalib.__version__

    from .dataset import Dataset
    from .engine import AnomalibEngine as Engine
    from .model import get_model

    __all__ = ["Dataset", "Engine", "get_model"]

except ImportError:
    AVAILABLE = False
