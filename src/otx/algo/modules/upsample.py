# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations
import inspect

import torch.nn as nn
from mmengine.registry import MODELS


MODELS.register_module('nearest', module=nn.Upsample)
MODELS.register_module('bilinear', module=nn.Upsample)


def build_upsample_layer(cfg: dict, *args, **kwargs) -> nn.Module:
    """Build upsample layer.

    Args:
        cfg (dict): The upsample layer config, which should contain:

            - type (str): Layer type.
            - scale_factor (int): Upsample ratio, which is not applicable to
              deconv.
            - layer args: Args needed to instantiate a upsample layer.
        args (argument list): Arguments passed to the ``__init__``
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the
            ``__init__`` method of the corresponding conv layer.

    Returns:
        nn.Module: Created upsample layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        raise KeyError(
            f'the cfg dict must contain the key "type", but got {cfg}')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')

    if inspect.isclass(layer_type):
        upsample = layer_type
    # Switch registry to the target scope. If `upsample` cannot be found
    # in the registry, fallback to search `upsample` in the
    # mmengine.MODELS.
    else:
        with MODELS.switch_scope_and_registry(None) as registry:
            upsample = registry.get(layer_type)
        if upsample is None:
            raise KeyError(f'Cannot find {upsample} in registry under scope '
                           f'name {registry.scope}')
        if upsample is nn.Upsample:
            cfg_['mode'] = layer_type
    layer = upsample(*args, **kwargs, **cfg_)
    return layer
