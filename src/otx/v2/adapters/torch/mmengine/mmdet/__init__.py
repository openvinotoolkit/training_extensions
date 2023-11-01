"""OTX adapters.torch.mmengine.mmdet module."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


AVAILABLE = True
VERSION = None
DEBUG = None

try:
    import mmdet

    VERSION = mmdet.__version__
    from mmdet.utils import register_all_modules

    register_all_modules(init_default_scope=True)

    from . import modules
    from .dataset import Dataset
    from .engine import MMDetEngine as Engine
    from .model import get_model, list_models

    __all__ = ["get_model", "Dataset", "Engine", "list_models", "modules"]

except ImportError as e:
    AVAILABLE = False
    DEBUG = e
