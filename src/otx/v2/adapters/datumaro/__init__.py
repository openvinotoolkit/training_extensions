"""OTX adapters.datumaro module."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


AVAILABLE = True
VERSION = None
DEBUG = None

try:
    import datumaro

    VERSION = datumaro.__version__
except ImportError as e:
    AVAILABLE = False
    DEBUG = e
