"""OTX adapters.datumaro module."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


AVAILABLE = True
VERSION = None

try:
    import datumaro  # noqa: F401

    VERSION = datumaro.__version__
except ImportError:
    AVAILABLE = False
