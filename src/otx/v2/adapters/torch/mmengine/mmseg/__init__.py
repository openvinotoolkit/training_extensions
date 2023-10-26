"""OTX adapters.torch.mmengine.mmseg module."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


AVAILABLE = True
VERSION = None
DEBUG = None

try:
    import mmseg

    VERSION = mmseg.__version__
    from mmseg.utils import register_all_modules

    register_all_modules(init_default_scope=True)

    from .mmseg import Dataset
    from .mmseg import MMPTEngine as Engine
    from .mmseg import get_model, list_models

    __all__ = ["get_model", "Dataset", "Engine", "list_models"]

except ImportError as e:
    AVAILABLE = False
    DEBUG = e
