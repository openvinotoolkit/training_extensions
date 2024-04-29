# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

"""This implementation replaces the functionality of mmcv.cnn.bricks.conv.build_conv_layer."""

from __future__ import annotations

import inspect

from torch import nn

CONV_DICT = {
    "Conv1d": nn.Conv1d,
    "Conv2d": nn.Conv2d,
    "Conv3d": nn.Conv3d,
    "Conv": nn.Conv2d,
}


def build_conv_layer(cfg: dict | None, *args, **kwargs) -> nn.Module:
    """Build convolution layer.

    Args:
        cfg (None or dict): The conv layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an conv layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding conv layer.

    Returns:
        nn.Module: Created conv layer.
    """
    if cfg is None:
        cfg_ = {"type": "Conv2d"}
    else:
        if not isinstance(cfg, dict):
            msg = "cfg must be a dict"
            raise TypeError(msg)
        if "type" not in cfg:
            msg = 'the cfg dict must contain the key "type"'
            raise KeyError(msg)
        cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")
    if inspect.isclass(layer_type):
        return layer_type(*args, **kwargs, **cfg_)
    conv_layer = CONV_DICT.get(layer_type)
    if conv_layer is None:
        msg = f"Cannot find {conv_layer} in {CONV_DICT.keys()}"
        raise KeyError(msg)
    return conv_layer(*args, **kwargs, **cfg_)
