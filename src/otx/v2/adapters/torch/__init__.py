"""OTX adapters.torch module."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


AVAILABLE = True
VERSION = None

try:
    import torch  # noqa: F401

    VERSION = torch.__version__
except ImportError:
    AVAILABLE = False
