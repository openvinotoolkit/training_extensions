"""OTX adapters.torch.mmengine.mmpretrain.modules."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Modules in mmX must be imported at least once before they can be used,
# as they must be registered in the registry.
# So these all need to be imported at runtime so that the module can be found in mmX.
from .datasets import *  # noqa: F403
from .models import *  # noqa: F403
from .optimizer import *  # noqa: F403
