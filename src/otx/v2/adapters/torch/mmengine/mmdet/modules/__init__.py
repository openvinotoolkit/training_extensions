"""OTX adapters.torch.mmengine.mmdet modules."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Modules in mmX must be imported at least once before they can be used,
# as they must be registered in the registry.
# So these all need to be imported at runtime so that the module can be found in mmX.
from . import evaluation, models

__all__ = ["dataset", "models", "evaluation"]
