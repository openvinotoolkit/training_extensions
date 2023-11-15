"""Collections of util functions related to torch."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Dict, Union

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
    def __init__(self, model: torch.nn.Module, enabled: bool, save_dir: Union[Path, str], max_iters: int = 1):
        """Helps to debug model and saves model weights, forward output and gradients during given maximum number of iterations.

        Args:
            model (Module): model to train and debug.
            save_dir (Path, str): path where to save ".pth" tensors
            max_iters (int): maximum number of iterations during which is needed to save the meta info of the model.
        """
        self._model = model
        self.enabled = enabled
        self.iter = 0
        self.max_iters = max_iters
        self._save_dir = Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._forward_output = {}
        self._gradient = {}
        if self.enabled:
            self._register_hook()

    def __enter__(self):
        if not self.enabled:
            return
        if self.iter < self.max_iters:
            torch.save(self._model.state_dict(), self._save_dir / f"model_weights_iter_{self.iter}.pth")

    def __call__(self, iter):
        self.iter = iter
        return self

    def __exit__(self, *args, **kwargs):
        if not self.enabled:
            return
        if self.iter < self.max_iters:
            if self._forward_output:
                torch.save(self._forward_output, self._save_dir / f"forward_output_per_layer_iter_{self.iter}.pth")

            for name, params in self._model.named_parameters():
                if params.grad is not None:
                    self.save_tensor_to_dict(self._gradient, name, params.grad)

            if self._gradient:
                torch.save(self._gradient, self._save_dir / f"gradients_per_layer_iter_{self.iter}.pth")

    def _register_hook(self):
        for name, layer in self._model.named_modules():
            layer.__name__ = name
            layer.register_forward_hook(self.save_outputs_hook)

    def save_outputs_hook(self, module, _, output):
        if self.iter < self.max_iters:
            self.save_tensor_to_dict(self._forward_output, module.__name__, output)

    @staticmethod
    def save_tensor_to_dict(dict_val: Dict, name: str, tensor):
        name_suffix = ""
        layer_idx = 0
        while name + name_suffix in dict_val:
            layer_idx += 1
            name_suffix = f"#{layer_idx}"

        dict_val[name + name_suffix] = tensor
