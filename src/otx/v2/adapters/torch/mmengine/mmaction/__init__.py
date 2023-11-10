"""OTX adapters.torch.mmengine.mmaction module."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


AVAILABLE = True
VERSION = None
DEBUG = None

try:
    import mmaction

    VERSION = mmaction.__version__
    from mmaction.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    
    import os
    os.environ["FEATURE_FLAGS_OTX_ACTION_TASKS"] = "1"

    from .dataset import MMActionDataset as Dataset
    from .engine import MMActionEngine as Engine
    from .model import get_model, list_models

    __all__ = ["get_model", "Dataset", "Engine", "list_models"]

except ImportError as e:
    AVAILABLE = False
    DEBUG = e
