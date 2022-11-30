"""OTX Adapters - mmaction2."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .data import OTXRawframeDataset
from .utils import export_model, patch_config, set_data_classes

__all__ = ["OTXRawframeDataset", "patch_config", "set_data_classes", "export_model"]
