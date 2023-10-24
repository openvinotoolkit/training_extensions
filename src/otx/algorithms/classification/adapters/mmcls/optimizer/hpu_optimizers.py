"""Optimizers for HPU."""

import torch
import habana_frameworks.torch.hpex.optimizers as hoptimizers
from typing import List
import inspect
from mmcv.runner import OPTIMIZERS


def register_habana_optimizers() -> List:
    habana_optimizers = []
    for module_name in dir(hoptimizers):
        if module_name.startswith('__'):
            continue
        _optim = getattr(hoptimizers, module_name)
        if inspect.isclass(_optim) and issubclass(_optim, torch.optim.Optimizer):
            OPTIMIZERS.register_module()(_optim)
            habana_optimizers.append(module_name)
    return habana_optimizers
