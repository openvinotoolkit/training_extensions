"""OTX adapters.torch.mmengine module."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


AVAILABLE = True
VERSION = None
DEBUG = None

try:
    import mmengine

    from . import modules

    VERSION = mmengine.__version__
    __all__ = ["modules"]
except ImportError as e:
    AVAILABLE = False
    DEBUG = e
