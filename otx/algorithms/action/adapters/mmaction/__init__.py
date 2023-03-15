"""OTX Adapters - mmaction2."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .data import OTXActionClsDataset, OTXActionDetDataset
from .models import register_action_backbones
from .utils import Exporter, patch_config, set_data_classes

__all__ = ["OTXActionClsDataset", "OTXActionDetDataset", "patch_config", "set_data_classes", "Exporter"]

register_action_backbones()
