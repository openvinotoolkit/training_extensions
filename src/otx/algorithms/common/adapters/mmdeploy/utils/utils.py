"""Util functions of otx.algorithms.common.adapters.mmdeploy."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from collections.abc import MutableMapping

import numpy as np
import torch


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


def numpy_2_list(data):
    """Converts NumPy arrays to Python lists."""

    if isinstance(data, np.ndarray):
        return data.tolist()

    if isinstance(data, MutableMapping):
        for key, value in data.items():
            data[key] = numpy_2_list(value)
    elif isinstance(data, (list, tuple)):
        data_ = []
        for value in data:
            data_.append(numpy_2_list(value))
        if isinstance(data, tuple):
            data = tuple(data_)
        else:
            data = data_
    return data
