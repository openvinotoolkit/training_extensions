# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os.path as osp
import platform
import shutil
import sys
import tempfile
import warnings
from importlib import import_module
from typing import Any, Callable, Union

from mmcv.utils import Config, ConfigDict
from mmcv.utils.config import BASE_KEY, DEPRECATION_KEY
from mmcv.utils.misc import import_modules_from_strings
from mmcv.utils.path import check_file_exist

from otx.mpa.utils.logger import get_logger

logger = get_logger()


class MPAConfig(Config):
    @staticmethod
    def _file2dict(filename, use_predefined_variables=True):
        filename = osp.abspath(osp.expanduser(filename))
        check_file_exist(filename)
        fileExtname = osp.splitext(filename)[1]
        if fileExtname not in [".py", ".json", ".yaml", ".yml"]:
            raise IOError("Only py/yml/yaml/json type are supported now!")

        with tempfile.TemporaryDirectory() as temp_config_dir:
            temp_config_file = tempfile.NamedTemporaryFile(dir=temp_config_dir, suffix=fileExtname)
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
            # close temp file
            temp_config_file.close()

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

            cfg_dict_list = list()
            cfg_text_list = list()
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
        cfg_dict, cfg_text = MPAConfig._file2dict(filename, use_predefined_variables)
        if import_custom_modules and cfg_dict.get("custom_imports", None):
            import_modules_from_strings(**cfg_dict["custom_imports"])
        return Config(cfg_dict, cfg_text=cfg_text, filename=filename)


def copy_config(cfg):
    if not isinstance(cfg, Config):
        ValueError(f"cannot copy this instance {type(cfg)}")
    # new_cfg = copy.deepcopy(cfg)
    # new_cfg._cfg_dict = copy.deepcopy(cfg._cfg_dict)
    # new_cfg.filename = cfg.filename
    import pickle

    data = pickle.dumps(cfg)
    return pickle.loads(data)


def update_or_add_custom_hook(cfg: Config, hook_cfg: ConfigDict):
    """Update hook cfg if same type is in custom_hook or append it"""
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
    """Remove hook cfg if hook_type is in custom_hook"""
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
    for k, v in list(cfg.items()):
        if isinstance(v, dict):
            recursively_update_cfg(v, criterion, update_dict)
        if criterion(k, v):
            cfg.update(update_dict)


def add_custom_hook_if_not_exists(cfg: Config, hook_cfg: ConfigDict):
    custom_hooks = cfg.get("custom_hooks", [])
    found = False
    for hook in custom_hooks:
        if hook["type"] == hook_cfg["type"]:
            found = True
            break
    if not found:
        custom_hooks.append(hook_cfg)
        cfg["custom_hooks"] = custom_hooks
