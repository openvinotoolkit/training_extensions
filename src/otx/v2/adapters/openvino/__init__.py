"""Module for otx.v2.adapters.openvino."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

AVAILABLE = True
VERSION = None
DEBUG = None

try:
    import openvino  # noqa: F401

    # flake8: noqa
    from .graph import *
    from .models import *
    from .ops import *

except ImportError as e:
    AVAILABLE = False
    DEBUG = e
