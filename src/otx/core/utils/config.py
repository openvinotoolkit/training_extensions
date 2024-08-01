# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions for config files."""

from __future__ import annotations

from numbers import Number
from typing import TYPE_CHECKING, Literal

from omegaconf import DictConfig, OmegaConf

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
    cfg: DictConfig | dict,
    to: Literal["tuple", "list"] = "tuple",
) -> MMConfig:
    """Convert OTX format config object to MMEngine config object."""
    from mmengine.config import Config as MMConfig

    cfg = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(cfg)
    dict_cfg = OmegaConf.to_container(cfg)

    if to == "tuple":
        return MMConfig(cfg_dict=to_tuple(dict_cfg))
    if to == "list":
        return MMConfig(cfg_dict=to_list(dict_cfg))

    raise ValueError(to)
