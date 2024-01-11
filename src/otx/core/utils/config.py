# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions for config files."""

from __future__ import annotations

from numbers import Number
from typing import TYPE_CHECKING, Any, Literal

from omegaconf import DictConfig, ListConfig, OmegaConf

if TYPE_CHECKING:
    from mmengine.config import Config as MMConfig


def to_tuple(dict_: dict) -> dict:
    """Find and replace tuple or list values in dict to tuple recursively."""
    # MMDET Mosaic asserts if "img_shape" is not tuple
    # File "mmdet/datasets/transforms/transforms.py", line 2324, in __init__

    for k, v in dict_.items():
        if isinstance(v, (tuple, list)) and all(isinstance(elem, Number) for elem in v):
            dict_[k] = tuple(v)
        elif isinstance(v, dict):
            to_tuple(v)

    return dict_


def to_list(dict_: dict) -> dict:
    """Find and replace tuple or list values in dict to list recursively."""
    # MMDET FPN asserts if "in_channels" is not list
    # File "mmdet/models/necks/fpn.py", line 88, in __init__

    for k, v in dict_.items():
        if isinstance(v, (tuple, list)) and all(isinstance(elem, Number) for elem in v):
            dict_[k] = list(v)
        elif isinstance(v, dict):
            to_list(v)

    return dict_


def convert_conf_to_mmconfig_dict(
    cfg: DictConfig,
    to: Literal["tuple", "list"] = "tuple",
) -> MMConfig:
    """Convert OTX format config object to MMEngine config object."""
    from mmengine.config import Config as MMConfig

    dict_cfg = OmegaConf.to_container(cfg)

    if to == "tuple":
        return MMConfig(cfg_dict=to_tuple(dict_cfg))
    if to == "list":
        return MMConfig(cfg_dict=to_list(dict_cfg))

    raise ValueError(to)


def mmconfig_dict_to_dict(obj: MMConfig | list[MMConfig]) -> list | dict:
    """Convert MMEngine config object to Python dictionary."""
    if isinstance(obj, list):
        return [mmconfig_dict_to_dict(x) for x in obj]
    if hasattr(obj, "to_dict"):
        return {k: mmconfig_dict_to_dict(v) for k, v in obj.to_dict().items()}

    return obj


def inplace_num_classes(
    cfg: DictConfig | ListConfig | Any,  # noqa: ANN401
    num_classes: int,
) -> DictConfig | ListConfig | Any:  # noqa: ANN401
    """Inplace the number of classes values in a given config object.

    Args:
        cfg: Config object to inplace the number of classes values
        num_classes: Number of classes to inplace
    Returns:
        Inplaced config object
    """
    if isinstance(cfg, DictConfig):
        for key in cfg:
            if key == "num_classes" and isinstance(cfg[key], int):
                cfg[key] = num_classes
            else:
                cfg[key] = inplace_num_classes(cfg[key], num_classes)

    if isinstance(cfg, ListConfig):
        for idx in range(len(cfg)):
            cfg[idx] = inplace_num_classes(cfg[idx], num_classes)

    return cfg
