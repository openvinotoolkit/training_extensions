"""Adapters for mmdeploy."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

__all__ = []

AVAILABLE = True
VERSION = None
DEBUG = None
try:
    import platform

    if platform.system() not in ("Linux", "Windows"):
        raise ImportError("mmdeploy is only supports Windows and Linux.")

    import mmdeploy  # noqa: F401

    from .ops import squeeze__default

    __all__ = ["squeeze__default"]
    VERSION = mmdeploy.__version__
except ImportError as e:
    AVAILABLE = False
    DEBUG = e
