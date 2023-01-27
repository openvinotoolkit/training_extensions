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
from typing import Any, Dict, List, Tuple, Union

from mmcv import Config, ConfigDict

from otx.api.entities.datasets import DatasetEntity
from otx.api.utils.argument_checks import (
    DatasetParamTypeCheck,
    check_input_parameters_type,
)
from otx.mpa.utils.logger import get_logger

from ._config_utils_get_configs_by_keys import get_configs_by_keys
from ._config_utils_get_configs_by_pairs import get_configs_by_pairs

logger = get_logger()


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


@check_input_parameters_type()
def remove_from_configs_by_type(configs: List[ConfigDict], type_name: str):
    """Update & remove by type."""
    indices = []
    for i, config in enumerate(configs):
        type_name_ = config.get("type", None)
        if type_name_ == type_name:
            indices.append(i)
    for i in reversed(indices):
        configs.pop(i)


def update_config(
    config: Union[Config, ConfigDict],
    pairs: Dict[Tuple[Any, ...], Any],
):
    """Update configs by path as a key and value as a target."""
    for path, value in pairs.items():
        path_ = list(reversed(path))
        ptr = config
        key = None
        while path_:
            key = path_.pop()
            if isinstance(ptr, (Config, Mapping)):
                if key not in ptr:
                    ptr[key] = ConfigDict()
            elif isinstance(ptr, (list, tuple)):
                assert isinstance(key, int), f"{key} of {path} must be int for ({type(ptr)}: {ptr})"
                assert len(ptr) < key, f"{key} of {path} exceeds {len(ptr)}"
            if len(path_) == 0:
                ptr[key] = value
            ptr = ptr[key]


@check_input_parameters_type()
def get_dataset_configs(config: Union[Config, ConfigDict], subset: str) -> List[ConfigDict]:
    """Get 'datasets' configs."""
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
def patch_default_config(config: Config):
    """Patch default config."""
    if "runner" not in config:
        config.runner = ConfigDict({"type": "EpochBasedRunner"})
    if "log_config" not in config:
        config.log_config = ConfigDict()
    if "evaluation" not in config:
        config.evaluation = ConfigDict()
    if "checkpoint_config" not in config:
        config.checkpoint_config = ConfigDict({"type": "CheckpointHook", "interval": 1})


@check_input_parameters_type()
def patch_data_pipeline(config: Config, data_pipeline: str = ""):
    """Replace data pipeline to data_pipeline.py if it exist."""
    if os.path.isfile(data_pipeline):
        data_pipeline_cfg = Config.fromfile(data_pipeline)
        config.merge_from_dict(data_pipeline_cfg)
    else:
        raise FileNotFoundError(f"data_pipeline: {data_pipeline} not founded")


@check_input_parameters_type()
def patch_color_conversion(config: Config):
    """Patch color conversion."""
    assert "data" in config

    for cfg in get_configs_by_pairs(config.data, dict(type="Normalize")):
        to_rgb = False
        if "to_rgb" in cfg:
            to_rgb = cfg.to_rgb
        cfg.to_rgb = not bool(to_rgb)


@check_input_parameters_type()
def patch_runner(config: Config):
    """Patch runner."""
    assert "runner" in config

    # Check that there is no conflict in specification of number of training epochs.
    # Move global definition of epochs inside runner config.
    if "total_epochs" in config:
        if is_epoch_based_runner(config.runner):
            if config.runner.max_epochs != config.total_epochs:
                logger.warning("Conflicting declaration of training epochs number.")
            config.runner.max_epochs = config.total_epochs
        else:
            logger.warning(f"Total number of epochs set for an iteration based runner {config.runner.type}.")
        remove_from_config(config, "total_epochs")

    # Change runner's type.
    if is_epoch_based_runner(config.runner):
        logger.info(f"Replacing runner from {config.runner.type} to EpochRunnerWithCancel.")
        config.runner.type = "EpochRunnerWithCancel"
    else:
        logger.info(f"Replacing runner from {config.runner.type} to IterBasedRunnerWithCancel.")
        config.runner.type = "IterBasedRunnerWithCancel"


@check_input_parameters_type()
def align_data_config_with_recipe(data_config: ConfigDict, config: Union[Config, ConfigDict]):
    """Align data_cfg with recipe_cfg."""
    # we assumed config has 'otx_dataset' and 'labels' key in it
    # by 'patch_datasets' function

    data_config = data_config.data
    config = config.data
    for subset in data_config.keys():
        subset_config = data_config.get(subset, {})
        for key in list(subset_config.keys()):
            found_config = get_configs_by_keys(config.get(subset), key, return_path=True)
            assert len(found_config) == 1
            value = subset_config.pop(key)
            path = list(found_config.keys())[0]
            update_config(subset_config, {path: value})


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
