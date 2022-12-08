"""Utils for common OTX algorithms."""

# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import copy
import glob
import os
import tempfile
from collections.abc import Mapping
from typing import Union, List, Any, Dict, Tuple

from mmcv import Config, ConfigDict

from otx.api.entities.datasets import DatasetEntity
from otx.api.utils.argument_checks import (
    DatasetParamTypeCheck,
    check_input_parameters_type,
)


@check_input_parameters_type()
def remove_from_config(config: Union[Config, ConfigDict], key: str):
    """Update & Remove configs."""
    if key in config:
        if isinstance(config, Config):
            del config._cfg_dict[key]  # pylint: disable=protected-access
        elif isinstance(config, ConfigDict):
            del config[key]
        else:
            raise ValueError(f"Unknown config type {type(config)}")


def remove_from_configs_by_type(configs: List[ConfigDict], type: str):
    """Update & remove by type"""
    indices = []
    for i, config in enumerate(configs):
        type_ = config.get("type", None)
        if type_ == type:
            indices.append(i)
    for i in reversed(indices):
        configs.pop(i)


def get_configs_by_dict(  # noqa: C901
    config: Union[Config, ConfigDict],
    pairs: Dict[Any, Any],
    *,
    return_path: bool = False,
) -> Union[List[ConfigDict], Dict[Tuple[Any, ...], ConfigDict]]:
    # TODO: multiple instance

    def get_config(config, path=()):
        out = dict()
        if isinstance(config, (Config, Mapping)):
            if all(
                [
                    True if config.get(key, None) == value else False
                    for key, value in pairs.items()
                ]
            ):
                return {path: config}
            for key, value in config.items():
                out.update(get_config(value, (*path, key)))
        elif isinstance(config, (list, tuple)):
            for idx, value in enumerate(config):
                out.update(get_config(value, (*path, idx)))
        return out

    out = get_config(config)
    if return_path:
        return out

    out_ = []
    for found in out.values():
        if isinstance(found, (list, tuple)):
            out_.extend(found)
        else:
            out_.append(found)
    return out_


def get_configs_by_keys(  # noqa: C901
    config: Union[Config, ConfigDict],
    keys: Union[Any, List[Any]],
    *,
    return_path: bool = False,
) -> Union[List[ConfigDict], Dict[Tuple[Any, ...], ConfigDict]]:
    """Return a list of configs based on key."""

    if not isinstance(keys, list):
        keys = [keys]

    def get_config(config, path=()):
        if path and path[-1] in keys:
            return {path: config}

        out = dict()
        if isinstance(config, (Config, Mapping)):
            for key, value in config.items():
                out.update(get_config(value, (*path, key)))
        elif isinstance(config, (list, tuple)):
            for idx, value in enumerate(config):
                out.update(get_config(value, (*path, idx)))
        return out

    out = get_config(config)
    if return_path:
        return out

    out_ = []
    for found in out.values():
        if isinstance(found, (list, tuple)):
            out_.extend(found)
        else:
            out_.append(found)
    return out_


def update_config(
    config: Union[Config, ConfigDict],
    update_config: Dict[Tuple[Any, ...], Any],
):
    for path, value in update_config.items():
        path_ = list(reversed(path))
        ptr = config
        key = None
        while path_:
            key = path_.pop()
            if isinstance(ptr, (Config, Mapping)):
                if key not in ptr:
                    ptr[key] = ConfigDict()
            elif isinstance(ptr, (list, tuple)):
                assert isinstance(key, int), (
                    f"{key} of {path} must be int for ({type(ptr)}: {ptr})"
                )
                assert len(ptr) < key, f"{key} of {path} exceeds {len(ptr)}"
            if len(path_) == 0:
                ptr[key] = value
            ptr = ptr[key]


@check_input_parameters_type()
def get_dataset_configs(
    config: Union[Config, ConfigDict], subset: str = "train"
) -> List[ConfigDict]:
    if config.data.get(subset, None) is None:
        return []
    data_cfg = config.data[subset]
    data_cfgs = get_configs_by_keys(data_cfg, ["dataset", "datasets"])
    return data_cfgs if data_cfgs else [data_cfg]


@check_input_parameters_type({"dataset": DatasetParamTypeCheck})
def prepare_for_testing(config: Union[Config, ConfigDict], dataset: DatasetEntity) -> Config:
    """Prepare configs for testing phase."""
    config = copy.deepcopy(config)
    # FIXME. Should working directories be modified here?
    config.data.test.otx_dataset = dataset
    return config


@check_input_parameters_type()
def is_epoch_based_runner(runner_config: ConfigDict):
    """Check Epoch based or Iter based runner."""
    return "Epoch" in runner_config.type


@check_input_parameters_type()
def config_from_string(config_string: str) -> Config:
    """Generate an mmcv config dict object from a string.

    :param config_string: string to parse
    :return config: configuration object
    """
    with tempfile.NamedTemporaryFile("w", suffix=".py") as temp_file:
        temp_file.write(config_string)
        temp_file.flush()
        return Config.fromfile(temp_file.name)


@check_input_parameters_type()
def patch_data_pipeline(config: Config, base_dir: str):
    """Replace data pipeline to data_pipeline.py if it exist."""
    data_pipeline_path = os.path.join(base_dir, "data_pipelne.py")
    if os.path.exists(data_pipeline_path):
        data_pipeline_cfg = Config.fromfile(data_pipeline_path)
        config.merge_from_dcit(data_pipeline_cfg)


def patch_color_conversion(pipeline):
    """Default data format for OTX is RGB, while mmx uses BGR, so negate the color conversion flag."""
    for pipeline_step in pipeline:
        if pipeline_step.type == "Normalize":
            to_rgb = False
            if "to_rgb" in pipeline_step:
                to_rgb = pipeline_step.to_rgb
            to_rgb = not bool(to_rgb)
            pipeline_step.to_rgb = to_rgb
        elif pipeline_step.type == "MultiScaleFlipAug":
            patch_color_conversion(pipeline_step.transforms)


DEFAULT_META_KEYS = (
    "filename",
    "ori_filename",
    "ori_shape",
    "img_shape",
    "pad_shape",
    "scale_factor",
    "flip",
    "flip_direction",
    "img_norm_cfg",
)


def get_meta_keys(pipeline_step):
    """Update meta_keys for ignore_labels."""
    meta_keys = list(pipeline_step.get("meta_keys", DEFAULT_META_KEYS))
    meta_keys.append("ignored_labels")
    pipeline_step["meta_keys"] = set(meta_keys)
    return pipeline_step


@check_input_parameters_type()
def prepare_work_dir(config: Union[Config, ConfigDict]) -> str:
    """Prepare configs of working directory."""
    base_work_dir = config.work_dir
    checkpoint_dirs = glob.glob(os.path.join(base_work_dir, "checkpoints_round_*"))
    train_round_checkpoint_dir = os.path.join(base_work_dir, f"checkpoints_round_{len(checkpoint_dirs)}")
    os.makedirs(train_round_checkpoint_dir)
    config.work_dir = train_round_checkpoint_dir
    if "meta" not in config.runner:
        config.runner.meta = ConfigDict()
    config.runner.meta.exp_name = f"train_round_{len(checkpoint_dirs)}"
    return train_round_checkpoint_dir


@check_input_parameters_type()
def get_data_cfg(config: Union[Config, ConfigDict], subset: str = "train") -> Config:
    """Return dataset configs."""
    data_cfg = config.data[subset]
    while "dataset" in data_cfg:
        data_cfg = data_cfg.dataset
    return data_cfg
