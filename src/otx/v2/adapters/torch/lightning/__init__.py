"""Lightning based Adapters."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

AVAILABLE = True
VERSION = None
DEBUG = None

try:
    import pytorch_lightning

    VERSION = pytorch_lightning.__version__
except ImportError as e:
    AVAILABLE = False
    DEBUG = e
