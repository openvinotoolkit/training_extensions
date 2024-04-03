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


def sync_batchnorm_2_batchnorm(module, dim=2):
    """Syncs the BatchNorm layers in a model to use regular BatchNorm layers."""
    if dim == 1:
        bn = torch.nn.BatchNorm1d
    elif dim == 2:
        bn = torch.nn.BatchNorm2d
    elif dim == 3:
        bn = torch.nn.BatchNorm3d
    else:
        raise NotImplementedError()

    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = bn(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad

        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig

    for name, child in module.named_children():
        module_output.add_module(name, sync_batchnorm_2_batchnorm(child, dim))

    del module

    return module_output
