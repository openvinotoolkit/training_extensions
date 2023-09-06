"""Adapters for mmdeploy."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

__all__ = []

AVAILABLE = True
VERSION = None
DEBUG = None
try:
    import mmdeploy  # noqa: F401

    from .ops import squeeze__default

    __all__.append("squeeze__default")
    VERSION = mmdeploy.__version__
except ImportError as e:
    AVAILABLE = False
    DEBUG = e
