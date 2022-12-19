"""OTX Adapters - mmaction2."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .data import OTXRawframeDataset
from .models import register_action_backbones
from .utils import patch_config, set_data_classes

__all__ = ["OTXRawframeDataset", "patch_config", "set_data_classes"]

register_action_backbones()
