"""A file for a function get_configs_by_keys()."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# NOTE: a workaround for https://github.com/python/mypy/issues/5028

from collections.abc import Mapping
from typing import Any, Dict, List, Literal, Sequence, Tuple, Union, overload

from mmcv import Config, ConfigDict


@overload
def get_configs_by_keys(
    configs: Union[Config, ConfigDict, Sequence[Config], Sequence[ConfigDict]],
    keys: Union[Any, List[Any]],
    *,
    return_path: Literal[True],
) -> Dict[Tuple[Any, ...], ConfigDict]:
    ...


@overload
def get_configs_by_keys(
    configs: Union[Config, ConfigDict, Sequence[Config], Sequence[ConfigDict]],
    keys: Union[Any, List[Any]],
    *,
    return_path: Literal[False] = False,
) -> List[ConfigDict]:
    ...


@overload
def get_configs_by_keys(
    configs: Union[Config, ConfigDict, Sequence[Config], Sequence[ConfigDict]],
    keys: Union[Any, List[Any]],
    *,
    return_path: bool,
) -> Union[List[ConfigDict], Dict[Tuple[Any, ...], ConfigDict]]:
    ...


def get_configs_by_keys(  # noqa: C901
    configs: Union[Config, ConfigDict, Sequence[Config], Sequence[ConfigDict]],
    keys: Union[Any, List[Any]],
    *,
    return_path: bool = False,
) -> Union[List[ConfigDict], Dict[Tuple[Any, ...], ConfigDict]]:
    """Get a list of configs by keys."""

    if not isinstance(keys, list):
        keys = [keys]

    def get_config(config, path=()):
        if path and path[-1] in keys:
            return {path: config}

        out = {}
        if isinstance(config, (Config, Mapping)):
            for key, value in config.items():
                out.update(get_config(value, (*path, key)))
        elif isinstance(config, (list, tuple)):
            for idx, value in enumerate(config):
                out.update(get_config(value, (*path, idx)))
        return out

    out = get_config(configs)
    if return_path:
        return out

    out_: List[ConfigDict] = []
    for found in out.values():
        if isinstance(found, (list, tuple)):
            out_.extend(found)
        else:
            out_.append(found)
    return out_
