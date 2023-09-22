"""Adapters for mmdeploy."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

AVAILABLE = True
VERSION = None
DEBUG = None
try:
    import platform

    if platform.system() not in ("Linux", "Windows"):
        raise ImportError("mmdeploy is only supports Windows and Linux.")

    import mmdeploy

    VERSION = mmdeploy.__version__
except ImportError as e:
    AVAILABLE = False
    DEBUG = e
