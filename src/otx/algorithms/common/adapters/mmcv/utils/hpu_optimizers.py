"""Optimizers for HPU."""

import torch
from typing import List
import inspect
from mmcv.runner import OPTIMIZERS

try:
    import habana_frameworks.torch.hpex.optimizers as hoptimizers
except ImportError:
    hoptimizers = None


def register_habana_optimizers() -> List:
    if hoptimizers is None:
        return []

    habana_optimizers = []
    for module_name in dir(hoptimizers):
        if module_name.startswith('__'):
            continue
        _optim = getattr(hoptimizers, module_name)
        if inspect.isclass(_optim) and issubclass(_optim, torch.optim.Optimizer):
            OPTIMIZERS.register_module()(_optim)
            habana_optimizers.append(module_name)
    return habana_optimizers

HABANA_OPTIMIZERS = register_habana_optimizers()
