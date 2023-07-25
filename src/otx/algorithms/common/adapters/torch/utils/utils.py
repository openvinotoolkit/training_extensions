"""Collections of util functions related to torch."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.nn import Module

try:
    from timm.models.layers import convert_sync_batchnorm as timm_cvt_sycnbn
except ImportError:
    timm_cvt_sycnbn = None


def model_from_timm(model: Module) -> bool:
    """Check a model comes from timm module.

    Args:
        model (Module): model to check it comes from timm module.

    Returns:
        bool : whether model comes from timm or not.
    """
    if "timm" in model.__module__.split("."):
        return True

    is_fisrt = True
    for sub_module in model.modules():
        if is_fisrt:  # First module is the module itself.
            is_fisrt = False
            continue

        if model_from_timm(sub_module):
            return True

    return False


def convert_sync_batchnorm(model: Module):
    """Convert BatchNorm layers to SyncBatchNorm layers.

    Args:
        model (Module): model containing batchnorm layers.
    """
    if timm_cvt_sycnbn is not None and model_from_timm(model):
        timm_cvt_sycnbn(model)
    else:
        torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
