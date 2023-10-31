"""OTX adapters.torch.mmengine.mmpretrain module."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


AVAILABLE = True
VERSION = None
DEBUG = None

try:
    import mmpretrain

    VERSION = mmpretrain.__version__
    from mmpretrain.utils import register_all_modules

    register_all_modules(init_default_scope=True)

    from .dataset import MMPretrainDataset as Dataset
    from .engine import MMPTEngine as Engine
    from .model import get_model, list_models

    __all__ = ["get_model", "Dataset", "Engine", "list_models"]

except ImportError as e:
    AVAILABLE = False
    DEBUG = e
