"""Collections of util functions related to torch."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Union, Dict
from pathlib import Path

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


class ModelDebugger:
    def __init__(self, model: torch.nn.Module, save_dir: Union[Path, str]):
        self._model = model
        self._save_dir = Path(save_dir)
        self._forward_output = {}
        self._gradient = {}

    def __enter__(self):
        torch.save(self._model.state_dict(), self._save_dir / "init_model.pth")
        self._register_hook()

    def __exit__(self, *args, **kwargs):
        if self._forward_output:
            torch.save(self._forward_output, self._save_dir / "forward_output.pth")
        
        for name, params in self._model.named_parameters():
            self.save_tensor_to_dict(self._gradient, name, params.grad)

        if self._gradient:
            torch.save(self._gradient, self._save_dir / "grad.pth")

    def _register_hook(self):
        for name, layer in self._model.named_modules():
            layer.__name__ = name
            layer.register_forward_hook(self.save_outputs_hook)

    def save_outputs_hook(self, module, _, output):
        self.save_tensor_to_dict(self._forward_output, module.__name__, output)

    @staticmethod
    def save_tensor_to_dict(dict_val: Dict, name: str, tensor):
        name_suffix = ""
        layer_idx = 0
        while name + name_suffix in dict_val:
            layer_idx += 1
            name_suffix = f"#{layer_idx}"

        dict_val[name + name_suffix] = tensor
