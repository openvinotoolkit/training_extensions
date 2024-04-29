# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

"""This implementation replaces the functionality of mmcv.cnn.bricks.padding.build_padding_layer."""

import inspect

from torch import nn

PADDING_DICT = {
    "zero": nn.ZeroPad2d,
    "reflect": nn.ReflectionPad2d,
    "replicate": nn.ReplicationPad2d,
}


def build_padding_layer(cfg: dict, *args, **kwargs) -> nn.Module:
    """Build padding layer.

    Args:
        cfg (dict): The padding layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a padding layer.

    Returns:
        nn.Module: Created padding layer.
    """
    if not isinstance(cfg, dict):
        msg = "cfg must be a dict"
        raise TypeError(msg)
    if "type" not in cfg:
        msg = 'the cfg dict must contain the key "type"'
        raise KeyError(msg)

    cfg_ = cfg.copy()
    padding_type = cfg_.pop("type")
    if inspect.isclass(padding_type):
        return padding_type(*args, **kwargs, **cfg_)
    # Switch registry to the target scope. If `padding_layer` cannot be found
    # in the registry, fallback to search `padding_layer` in the
    # mmengine.MODELS.
    padding_layer = PADDING_DICT.get(padding_type)
    if padding_layer is None:
        msg = f"Cannot find {padding_layer} in {PADDING_DICT.keys()} "
        raise KeyError(msg)
    return padding_layer(*args, **kwargs, **cfg_)
