"""Optimizers for HPU."""

import inspect
from typing import List

import torch
from mmcv.runner import OPTIMIZERS

try:
    import habana_frameworks.torch.hpex.optimizers as hoptimizers
except ImportError:
    hoptimizers = None


def register_habana_optimizers() -> List:
    """Register habana optimizers."""
    if hoptimizers is None:
        return []

    habana_optimizers = []
    for module_name in dir(hoptimizers):
        if module_name.startswith("__"):
            continue
        _optim = getattr(hoptimizers, module_name)
        if inspect.isclass(_optim) and issubclass(_optim, torch.optim.Optimizer):
            OPTIMIZERS.register_module()(_optim)
            habana_optimizers.append(module_name)
    return habana_optimizers


HABANA_OPTIMIZERS = register_habana_optimizers()
