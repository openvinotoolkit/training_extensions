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
import os.path as osp
import platform
import shutil
import sys
import tempfile
import warnings
from collections.abc import Mapping
from importlib import import_module
from typing import Any, Callable, Dict, List, Tuple, Union

from mmcv import Config, ConfigDict
from mmcv.utils.config import BASE_KEY, DEPRECATION_KEY
from mmcv.utils.misc import import_modules_from_strings
from mmcv.utils.path import check_file_exist

from otx.algorithms.common.utils.logger import get_logger
from otx.api.entities.datasets import DatasetEntity
from otx.api.utils.argument_checks import (
    DatasetParamTypeCheck,
    check_input_parameters_type,
)

from ._config_utils_get_configs_by_keys import get_configs_by_keys
from ._config_utils_get_configs_by_pairs import get_configs_by_pairs

logger = get_logger()


# TODO: refactor Config
class MPAConfig(Config):
    """A class that extends the base `Config` class, adds additional functionality for loading configuration files."""

    @staticmethod
    def _file2dict(
        filename, use_predefined_variables=True
    ):  # pylint: disable=too-many-locals, too-many-branches, too-many-statements
        """Static method that loads the configuration file and returns a dictionary of its contents.

        :param filename: str, the path of the configuration file to be loaded.
        :param use_predefined_variables: bool, a flag indicating whether to substitute predefined variables in the
                                          configuration file.
        :return: tuple of dictionary and string. Returns a dictionary containing the contents of the configuration file
                 and a string representation of the configuration file.
        :raises: IOError if the file type is not supported.
        """
        filename = osp.abspath(osp.expanduser(filename))
        check_file_exist(filename)
        extender = osp.splitext(filename)[1]
        if extender not in [".py", ".json", ".yaml", ".yml"]:
            raise IOError("Only py/yml/yaml/json type are supported now!")

        with tempfile.TemporaryDirectory() as temp_config_dir:
            with tempfile.NamedTemporaryFile(dir=temp_config_dir, suffix=extender) as temp_config_file:
                if platform.system() == "Windows":
                    temp_config_file.close()
                temp_config_name = osp.basename(temp_config_file.name)
                # Substitute predefined variables
                if use_predefined_variables:
                    Config._substitute_predefined_vars(filename, temp_config_file.name)
                else:
                    shutil.copyfile(filename, temp_config_file.name)
                # Substitute base variables from placeholders to strings
                base_var_dict = Config._pre_substitute_base_vars(temp_config_file.name, temp_config_file.name)
                if filename.endswith(".py"):
                    temp_module_name = osp.splitext(temp_config_name)[0]
                    sys.path.insert(0, temp_config_dir)
                    Config._validate_py_syntax(filename)
                    mod = import_module(temp_module_name)
                    sys.path.pop(0)
                    cfg_dict = {name: value for name, value in mod.__dict__.items() if not name.startswith("__")}
                    # delete imported module
                    del sys.modules[temp_module_name]
                elif filename.endswith((".yml", ".yaml", ".json")):
                    import mmcv

                    cfg_dict = mmcv.load(temp_config_file.name)

        # check deprecation information
        if DEPRECATION_KEY in cfg_dict:
            deprecation_info = cfg_dict.pop(DEPRECATION_KEY)
            warning_msg = f"The config file {filename} will be deprecated " "in the future."
            if "expected" in deprecation_info:
                warning_msg += f' Please use {deprecation_info["expected"]} ' "instead."
            if "reference" in deprecation_info:
                warning_msg += " More information can be found at " f'{deprecation_info["reference"]}'
            warnings.warn(warning_msg)

        cfg_text = filename + "\n"
        with open(filename, "r", encoding="utf-8") as f:
            # Setting encoding explicitly to resolve coding issue on windows
            cfg_text += f.read()

        if BASE_KEY in cfg_dict:
            cfg_dir = osp.dirname(filename)
            base_filename = cfg_dict.pop(BASE_KEY)
            base_filename = base_filename if isinstance(base_filename, list) else [base_filename]

            cfg_dict_list = []
            cfg_text_list = []
            for f in base_filename:
                _cfg_dict, _cfg_text = MPAConfig._file2dict(osp.join(cfg_dir, f))
                cfg_dict_list.append(_cfg_dict)
                cfg_text_list.append(_cfg_text)

            base_cfg_dict = dict()
            # for c in cfg_dict_list:
            #     duplicate_keys = base_cfg_dict.keys() & c.keys()
            #     if len(duplicate_keys) > 0:
            #         raise KeyError('Duplicate key is not allowed among bases. '
            #                        f'Duplicate keys: {duplicate_keys}')
            #     base_cfg_dict.update(c)
            for c in cfg_dict_list:
                if len(base_cfg_dict.keys() & c.keys()) > 0:
                    # raise KeyError(f'Duplicate key is not allowed among bases [{base_cfg_dict.keys() & c.keys()}]')
                    logger.warning(f"Duplicate key is detected among bases [{base_cfg_dict.keys() & c.keys()}]")
                    logger.debug(f"base = {base_cfg_dict}, cfg = {c}")
                    base_cfg_dict = Config._merge_a_into_b(base_cfg_dict, c)
                    logger.debug(f"merged dict = {base_cfg_dict}")
                else:
                    base_cfg_dict.update(c)

            # Subtitute base variables from strings to their actual values
            cfg_dict = Config._substitute_base_vars(cfg_dict, base_var_dict, base_cfg_dict)

            base_cfg_dict = Config._merge_a_into_b(cfg_dict, base_cfg_dict)
            cfg_dict = base_cfg_dict

            # merge cfg_text
            cfg_text_list.append(cfg_text)
            cfg_text = "\n".join(cfg_text_list)

        return cfg_dict, cfg_text

    @staticmethod
    def fromfile(filename, use_predefined_variables=True, import_custom_modules=True):
        """Static method that loads a configuration file and returns an instance of `Config` class.

        :param filename: str, the path of the configuration file to be loaded.
        :param use_predefined_variables: bool, a flag indicating whether to substitute predefined variables in the
                                          configuration file.
        :param import_custom_modules: bool, a flag indicating whether to import custom modules.
        :return: Config object, an instance of `Config` class containing the contents of the configuration file.
        """
        cfg_dict, cfg_text = MPAConfig._file2dict(filename, use_predefined_variables)
        if import_custom_modules and cfg_dict.get("custom_imports", None):
            import_modules_from_strings(**cfg_dict["custom_imports"])
        return Config(cfg_dict, cfg_text=cfg_text, filename=filename)


def copy_config(cfg):
    """A function that creates a deep copy of the input configuration object.

    :param cfg: Config object, an instance of `Config` class to be copied.
    :return: Config object, a deep copy of the input configuration object.
    :raises: ValueError if the input object is not an instance of `Config` class.
    """
    if not isinstance(cfg, Config):
        raise ValueError(f"cannot copy this instance {type(cfg)}")
    # new_cfg = copy.deepcopy(cfg)
    # new_cfg._cfg_dict = copy.deepcopy(cfg._cfg_dict)
    # new_cfg.filename = cfg.filename
    import pickle

    data = pickle.dumps(cfg)
    return pickle.loads(data)


def update_or_add_custom_hook(cfg: Config, hook_cfg: ConfigDict):
    """Update hook cfg if same type is in custom_hook or append it."""
    custom_hooks = cfg.get("custom_hooks", [])
    custom_hooks_updated = False
    for custom_hook in custom_hooks:
        if custom_hook["type"] == hook_cfg["type"]:
            custom_hook.update(hook_cfg)
            custom_hooks_updated = True
            break
    if not custom_hooks_updated:
        custom_hooks.append(hook_cfg)
    cfg["custom_hooks"] = custom_hooks


def remove_custom_hook(cfg: Config, hook_type: str):
    """Remove hook cfg if hook_type is in custom_hook."""
    custom_hooks = cfg.get("custom_hooks", [])
    if len(custom_hooks) > 0:
        idx_to_del = None
        for i, custom_hook in enumerate(custom_hooks):
            if custom_hook["type"] == hook_type:
                idx_to_del = i
                break
        if idx_to_del is not None:
            del custom_hooks[idx_to_del]


def recursively_update_cfg(
    cfg: Union[Config, dict],
    criterion: Callable[[Any, Any], bool],
    update_dict: Any,
):
    """A function that recursively updates the input dictionary or `Config` object with a new dictionary.

    :param cfg: Union[Config, dict], an input dictionary or `Config` object to be updated.
    :param criterion: Callable[[Any, Any], bool], a function that determines whether to update a key-value pair based on
                      a criterion. The function takes two arguments: key and value, and returns a boolean.
    :param update_dict: Any, a dictionary to be used for updating the input dictionary.
    :return: None
    """
    for key, val in list(cfg.items()):
        if isinstance(val, dict):
            recursively_update_cfg(val, criterion, update_dict)
        if criterion(key, val):
            cfg.update(update_dict)


def add_custom_hook_if_not_exists(cfg: Config, hook_cfg: ConfigDict):
    """A function that adds a custom hook to the input `Config` object if it doesn't already exist.

    :param cfg: Config object, an instance of `Config` class to which the custom hook will be added.
    :param hook_cfg: ConfigDict object, an instance of `ConfigDict` class representing the custom hook to be added.
    :return: None
    """
    custom_hooks = cfg.get("custom_hooks", [])
    found = False
    for hook in custom_hooks:
        if hook["type"] == hook_cfg["type"]:
            found = True
            break
    if not found:
        custom_hooks.append(hook_cfg)
        cfg["custom_hooks"] = custom_hooks


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
    """A function that retrieves 'datasets' configurations from the input `Config` object or `ConfigDict` object.

    :param config: Union[Config, ConfigDict], an instance of `Config` class or `ConfigDict` class containing the
                   configurations.
    :param subset: str, a string representing the subset for which the 'datasets' configuration is required.
    :return: List[ConfigDict], a list of 'datasets' configuration dictionaries.
    """
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
    if is_epoch_based_runner(config.runner) and config.runner.type != "EpochRunnerWithCancel":
        logger.info(f"Replacing runner from {config.runner.type} to EpochRunnerWithCancel.")
        config.runner.type = "EpochRunnerWithCancel"
    elif not is_epoch_based_runner(config.runner) and config.runner.type != "IterBasedRunnerWithCancel":
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
