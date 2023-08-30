"""OTX adapters.torch.mmengine.mmpretrain module."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


AVAILABLE = True
VERSION = None

try:
    import mmpretrain  # noqa: F401

    VERSION = mmpretrain.__version__
    from mmpretrain.utils import register_all_modules

    register_all_modules(init_default_scope=True)

    from .dataset import Dataset
    from .engine import MMPTEngine as Engine
    from .model import get_model, list_models

    __all__ = ["get_model", "Dataset", "Engine", "list_models"]

except ImportError:
    AVAILABLE = False
