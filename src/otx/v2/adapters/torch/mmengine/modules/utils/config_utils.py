"""Utils for common OTX algorithms."""
# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import math
import copy
import multiprocessing
import os.path as osp
import platform
import shutil
import sys
import tempfile
import warnings
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
from enum import Enum
import torch
from mmengine.config import Config, ConfigDict
from mmengine.config.config import BASE_KEY, DEPRECATION_KEY
from mmengine.utils import check_file_exist, import_modules_from_strings
from torch.utils.data import DataLoader
import numpy as np
from otx.v2.api.utils.logger import get_logger

from ._config_utils_get_configs_by_pairs import get_configs_by_pairs

logger = get_logger()


class CustomConfig(Config):
    """A class that extends the base `Config` class, adds additional functionality for loading configuration files."""

    @staticmethod
    def _file2dict(
        filename: str,
        use_predefined_variables: bool = True,
    ) -> Tuple[Config, str]:
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
            raise OSError("Only py/yml/yaml/json type are supported now!")

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
                    import mmengine

                    cfg_dict = mmengine.load(temp_config_file.name)

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
        with open(filename, encoding="utf-8") as f:
            # Setting encoding explicitly to resolve coding issue on windows
            cfg_text += f.read()

        if BASE_KEY in cfg_dict:
            cfg_dir = osp.dirname(filename)
            base_key = cfg_dict.pop(BASE_KEY)
            base_filename: List[str] = base_key if isinstance(base_key, list) else [base_key]

            cfg_dict_list = []
            cfg_text_list = []
            for file_path in base_filename:
                _cfg_dict, _cfg_text = CustomConfig._file2dict(osp.join(cfg_dir, file_path))
                cfg_dict_list.append(_cfg_dict)
                cfg_text_list.append(_cfg_text)

            base_cfg_dict: Union[Config, Dict] = {}
            # for c in cfg_dict_list:
            #     if len(duplicate_keys) > 0:
            #         raise KeyError('Duplicate key is not allowed among bases. '
            #                        f'Duplicate keys: {duplicate_keys}')
            for c in cfg_dict_list:
                if len(base_cfg_dict.keys() & c.keys()) > 0:
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
    def fromfile(filename: str, use_predefined_variables: bool = True, import_custom_modules: bool = True) -> Config:
        """Static method that loads a configuration file and returns an instance of `Config` class.

        :param filename: str, the path of the configuration file to be loaded.
        :param use_predefined_variables: bool, a flag indicating whether to substitute predefined variables in the
                                          configuration file.
        :param import_custom_modules: bool, a flag indicating whether to import custom modules.
        :return: Config object, an instance of `Config` class containing the contents of the configuration file.
        """
        cfg_dict, cfg_text = CustomConfig._file2dict(filename, use_predefined_variables)
        if import_custom_modules and cfg_dict.get("custom_imports", None):
            import_modules_from_strings(**cfg_dict["custom_imports"])
        return CustomConfig(cfg_dict, cfg_text=cfg_text, filename=filename)

    @staticmethod
    def to_dict(config: Config | ConfigDict) -> dict:
        """Converts a Config object to a dictionary.

        Args:
            config(Config): The Config object to convert.

        Return:
            dict: The resulting dictionary.
        """
        output_dict = {}
        for key, value in config.items():
            if isinstance(value, (Config, ConfigDict)):
                output_dict[key] = CustomConfig.to_dict(value)
            else:
                output_dict[key] = value

        return output_dict

    @property
    def pretty_text(self) -> str:
        """Make python file human-readable.

        It's almost same as mmengine.Config's code but code to reformat using yapf is removed to reduce time.
        """

        indent = 4

        def _indent(s_: str, num_spaces: int) -> str:
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            _s = "\n".join(s)
            _s = first + "\n" + _s
            return _s

        def _format_basic_types(k: str, v: list, use_mapping: bool = False) -> str:
            v_str = f"'{v}'" if isinstance(v, str) else str(v)

            if use_mapping:
                k_str = f"'{k}'" if isinstance(k, str) else str(k)
                attr_str = f"{k_str}: {v_str}"
            else:
                attr_str = f"{str(k)}={v_str}"
            attr_str = _indent(attr_str, indent)

            return attr_str

        def _format_list(k: str, v: list, use_mapping: bool = False) -> str:
            # check if all items in the list are dict
            if all(isinstance(_, dict) for _ in v):
                v_str = "[\n"
                v_str += "\n".join(f"dict({_indent(_format_dict(v_), indent)})," for v_ in v).rstrip(",")
                if use_mapping:
                    k_str = f"'{k}'" if isinstance(k, str) else str(k)
                    attr_str = f"{k_str}: {v_str}"
                else:
                    attr_str = f"{str(k)}={v_str}"
                attr_str = _indent(attr_str, indent) + "]"
            else:
                attr_str = _format_basic_types(k, v, use_mapping)
            return attr_str

        def _contain_invalid_identifier(dict_str: dict) -> bool:
            contain_invalid_identifier = False
            for key_name in dict_str:
                contain_invalid_identifier |= not str(key_name).isidentifier()
            return contain_invalid_identifier

        def _format_dict(input_dict: dict, outest_level: bool = False) -> str:
            r = ""
            s = []

            use_mapping = _contain_invalid_identifier(input_dict)
            if use_mapping:
                r += "{"
            for idx, (k, v) in enumerate(input_dict.items()):
                is_last = idx >= len(input_dict) - 1
                end = "" if outest_level or is_last else ","
                if isinstance(v, dict):
                    v_str = "\n" + _format_dict(v)
                    if use_mapping:
                        k_str = f"'{k}'" if isinstance(k, str) else str(k)
                        attr_str = f"{k_str}: dict({v_str}"
                    else:
                        attr_str = f"{str(k)}=dict({v_str}"
                    attr_str = _indent(attr_str, indent) + ")" + end
                elif isinstance(v, list):
                    attr_str = _format_list(k, v, use_mapping) + end
                else:
                    attr_str = _format_basic_types(k, v, use_mapping) + end

                s.append(attr_str)
            r += "\n".join(s)
            if use_mapping:
                r += "}"
            return r

        cfg_dict = self._cfg_dict.to_dict()
        text = _format_dict(cfg_dict, outest_level=True)

        return text

    @staticmethod
    def merge_cfg_dict(base_dict: Union[Config, Dict], cfg_dict: Union[Config, Dict]) -> dict:
        if isinstance(base_dict, Config):
            base_dict = base_dict._cfg_dict.to_dict()
        if isinstance(cfg_dict, Config):
            cfg_dict = cfg_dict._cfg_dict.to_dict()
        return CustomConfig._merge_a_into_b(cfg_dict, base_dict)

    def dump(self, file: Optional[Union[str, Path]] = None) -> None:
        """Dump config to file or return config text.

        Args:
            file (str or Path, optional): If not specified, then the object
            is dumped to a str, otherwise to a file specified by the filename.
            Defaults to None.

        Returns:
            str or None: Config text.
        """
        # if file is None:
        #     if self.filename is None or self.filename.endswith('.py'):
        #     with open(file, 'w', encoding='utf-8') as f:


def update_or_add_custom_hook(cfg: Config, hook_cfg: ConfigDict) -> None:
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


WARN_MSG = (
    "The config used in the build is"
    "stored as an object in the configuration"
    "file because the object doesn't have it."
    "This can result in a non-reusable configs.py."
)


def dump_lazy_config(config: Config, file: Optional[Union[str, Path]] = None, scope: str = "mmengine") -> Config:
    # Dump Model(torch.nn.Module) Object to Dict
    output_config = copy.deepcopy(config)
    model = output_config.get("model", None)
    if isinstance(model, torch.nn.Module):
        model_config = {}
        if hasattr(model, "_config"):
            model_config = model._config
        else:
            logger.warning(WARN_MSG)
        model_config["type"] = f"{model.__class__.__qualname__}"
        model_config["_scope_"] = scope
        output_config["model"] = model_config
    # Dump Dataloader Object to Dict
    for subset in ["train", "val", "test"]:
        dataloader = output_config.get(f"{subset}_dataloader", None)
        if isinstance(dataloader, DataLoader):
            dl_config = {}
            if hasattr(dataloader, "configs"):
                dl_config = dataloader.configs
            else:
                logger.warning(WARN_MSG)
            dl_config["dataset"]["_scope_"] = scope
            output_config[f"{subset}_dataloader"] = dl_config
    if file is not None:
        # TODO: filename is not member of runner.__init__
        output_config.dump(file=file)

    return output_config


class InputSizePreset(Enum):
    """Configurable input size preset."""

    DEFAULT = "Default"
    AUTO = "Auto"
    _64x64 = "64x64"
    _128x128 = "128x128"
    _224x224 = "224x224"
    _256x256 = "256x256"
    _384x384 = "384x384"
    _512x512 = "512x512"
    _768x768 = "768x768"
    _1024x1024 = "1024x1024"

    @staticmethod
    def parse(value: str) -> Optional[Tuple[int, int]]:
        """Parse string value to tuple."""
        if value == "Default":
            return None
        if value == "Auto":
            return (0, 0)
        parsed_tocken = re.match("(\\d+)x(\\d+)", value)
        if parsed_tocken is None:
            return None
        return (int(parsed_tocken.group(1)), int(parsed_tocken.group(2)))

    @property
    def tuple(self) -> Optional[Tuple[int, int]]:
        """Returns parsed tuple."""
        return InputSizePreset.parse(self.value)

    @classmethod
    def input_sizes(cls) -> list:
        """Returns list of actual size tuples."""
        return [e.tuple for e in cls if e.value[0].isdigit()]


class InputSizeManager:
    """Class for changing input size and getting input size value by checking data pipeline.

    NOTE: "resize", "pad", "crop", "mosaic", "randomaffine", "multiscaleflipaug" , "AutoAugment" and "TwoCropTransform"
    are considered at now. If other data pipelines exist, it can work differently than expected.

    Args:
        config (Dict): Global configuration including data config w/ "train", "val" or "test" data pipeline.
        base_input_size (Optional[Union[int, List[int], Dict[str, Union[int, List[int]]]]], optional):
            Default input size. If it's a None, it's estimated based on data pipeline. If it's an integer,
            it's expected that all data pipeline have (base_input_size x base_input_size) input size.
            If it's an integer list, all data pipeline have same (base_input_size[0] x base_input_size[1]) input size.
            If it's dictionary, each data pipeline has specified input size. It should have format as below:
                {"train" : [w, h], "val" : [w, h], "test" : [w, h]}
    """

    PIPELINE_TO_CHANGE: Dict[str, List[str]] = {
        "resize": ["size", "img_scale"],
        "pad": ["size"],
        "crop": ["crop_size"],
        "mosaic": ["img_scale"],
        "randomaffine": ["border"],
        "multiscaleflipaug": ["img_scale"],
    }
    PIPELINE_WRAPPER: Dict[str, List[str]] = {
        "MultiScaleFlipAug": ["transforms"],
        "AutoAugment": ["policies"],
        "TwoCropTransform": ["view0", "view1", "pipeline"],
        "LoadResizeDataFromOTXDataset": ["resize_cfg"],
    }
    SUBSET_TYPES: Tuple[str, str, str, str] = ("train", "val", "test", "unlabeled")

    MIN_RECOGNIZABLE_OBJECT_SIZE = 32  # Minimum object size recognizable by NNs: typically 16 ~ 32
    # meaning NxN input pixels being downscaled to 1x1 on feature map
    MIN_DETECTION_INPUT_SIZE = 256  # Minimum input size for object detection

    def __init__(
        self,
        config: Dict,
        base_input_size: Optional[Union[int, Tuple[int, int], Dict[str, int], Dict[str, Tuple[int, int]]]] = None,
    ) -> None:
        self._config = config
        self._data_config = config.get("data", {})
        if isinstance(base_input_size, int):
            base_input_size = (base_input_size, base_input_size)
        elif isinstance(base_input_size, dict):
            for subset in base_input_size.keys():
                if isinstance(base_input_size[subset], int):
                    size = base_input_size[subset]
                    base_input_size[subset] = (size, size)  # type: ignore[assignment]
            for subset in self.SUBSET_TYPES:
                if subset in self._data_config and subset not in base_input_size:
                    raise ValueError(f"There is {subset} data configuration but base input size for it doesn't exists.")

        self._base_input_size = base_input_size

    def set_input_size(self, input_size: Union[int, List[int], Tuple[int, int]]) -> None:
        """Set input size in data pipe line.

        Args:
            input_size (Union[int, List[int]]):
                input size to set. If it's an integer, (input_size x input_size) will be set.
                If input_size is an integer list, (input_size[0] x input_size[1]) will be set.
        """
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        if not isinstance(self.base_input_size, dict):
            resize_ratio = (input_size[0] / self.base_input_size[0], input_size[1] / self.base_input_size[1])

        # Scale size values in data pipelines
        for subset in self.SUBSET_TYPES:
            if subset in self._data_config:
                if isinstance(self.base_input_size, dict):
                    resize_ratio = (
                        input_size[0] / self.base_input_size[subset][0],
                        input_size[1] / self.base_input_size[subset][1],
                    )
                pipelines = self._get_pipelines(subset)
                if isinstance(pipelines, dict):
                    # Deals with {"view0": [...], "view1": [...]}
                    for pipeline in pipelines.values():
                        self._set_pipeline_size_value(pipeline, resize_ratio)
                else:
                    self._set_pipeline_size_value(pipelines, resize_ratio)

        # Set model size
        model_cfg = self._config.get("model", {})
        model_cfg["input_size"] = input_size
        if model_cfg.get("type", "") == "CustomYOLOX":
            # - needed only for YOLOX
            if input_size[0] % 32 != 0 or input_size[1] % 32 != 0:
                raise ValueError("YOLOX should have input size being multiple of 32.")

    @property
    def base_input_size(self) -> Union[Tuple[int, int], Dict[str, Tuple[int, int]]]:
        """Getter function of `base_input_size` attirbute.

        If it isn't set when intializing class, it's estimated by checking data pipeline.
        Same value is returned after estimation.

        Raises:
            RuntimeError: If failed to estimate base input size from data pipeline, raise an error.

        Returns:
            Union[List[int], Dict[str, List[int]]]: Base input size.
        """
        if self._base_input_size is not None:
            return self._base_input_size  # type: ignore[return-value]

        input_size = self.get_input_size_from_cfg()
        if input_size is None:
            raise RuntimeError("There isn't any pipeline in the data configurations.")

        self._base_input_size = input_size
        return input_size

    def get_input_size_from_cfg(
        self, subset: Union[str, List[str]] = ["test", "val", "train"]
    ) -> Optional[Tuple[int, int]]:
        """Estimate image size using data pipeline.

        Args:
            subset (Union[str, List[str]], optional): Which pipelines to check. Defaults to ["test", "val", "train"].

        Returns:
            Union[None, List[int]]: Return estimiated input size. If failed to estimate, return None.
        """
        if isinstance(subset, str):
            subset = [subset]

        for target_subset in subset:
            if target_subset in self._data_config:
                input_size = self._estimate_post_img_size(self._data_config[target_subset]["pipeline"])
                if input_size is not None:
                    return tuple(input_size)  # type: ignore[return-value]

        return None

    def _estimate_post_img_size(
        self, pipelines: Union[Dict, List[Dict]], default_size: Optional[List[int]] = None
    ) -> Union[List[int], None]:
        # NOTE: Mosaic isn't considered in this step because Mosaic and following RandomAffine don't change image size
        post_img_size = default_size

        if isinstance(pipelines, dict):
            for pipeline in pipelines.values():
                # Deals with {"view0": [...], "view1": [...]}
                # Just using the first one to estimate
                return self._estimate_post_img_size(pipeline, post_img_size)

        for pipeline in pipelines:
            if "resize" in pipeline["type"].lower():
                img_size = self._get_size_value(pipeline, "resize")
                if img_size is not None:
                    post_img_size = img_size
            elif "pad" in pipeline["type"].lower():
                img_size = self._get_size_value(pipeline, "pad")
                if img_size is not None:
                    if post_img_size is None:
                        post_img_size = img_size
                    else:
                        for i in range(2):
                            if post_img_size[i] < img_size[i]:
                                post_img_size[i] = img_size[i]
            elif "crop" in pipeline["type"].lower():
                img_size = self._get_size_value(pipeline, "crop")
                if img_size is not None:
                    if post_img_size is None:
                        post_img_size = img_size
                    else:
                        for i in range(2):
                            if post_img_size[i] > img_size[i]:
                                post_img_size[i] = img_size[i]
            elif pipeline["type"] == "MultiScaleFlipAug":
                img_size = self._get_size_value(pipeline, "multiscaleflipaug")
                if img_size is not None:
                    post_img_size = img_size

        for pipeline_name, sub_pipeline_names in self.PIPELINE_WRAPPER.items():
            if pipeline_name == pipeline["type"]:
                for sub_pipeline_name in sub_pipeline_names:
                    if sub_pipeline_name in pipeline:
                        sub_pipeline = pipeline[sub_pipeline_name]
                        if isinstance(sub_pipeline, dict):
                            sub_pipeline = [sub_pipeline]
                        elif isinstance(sub_pipeline[0], list):
                            sub_pipeline = sub_pipeline[0]
                        post_img_size = self._estimate_post_img_size(sub_pipeline, post_img_size)
                        break

        return post_img_size

    @classmethod
    def _get_size_value(cls, pipeline: Dict, attr: str) -> Union[List[int], None]:
        for pipeline_attr in cls.PIPELINE_TO_CHANGE[attr]:
            if pipeline_attr not in pipeline:
                continue
            size_val = pipeline[pipeline_attr]
            if isinstance(size_val, int):
                return [size_val, size_val]
            elif isinstance(size_val, tuple):
                return list(size_val)
            elif isinstance(size_val, list):
                return list(size_val[0])

        return None

    def _get_pipelines(self, subset: str) -> list:
        if "pipeline" in self._data_config[subset]:
            return self._data_config[subset]["pipeline"]
        if "dataset" in self._data_config[subset]:
            return self._data_config[subset]["dataset"]["pipeline"]
        raise RuntimeError("Failed to find pipeline.")

    def _set_pipeline_size_value(
        self, pipeline: Union[Dict, List[Dict]], scale: Tuple[Union[int, float], Union[int, float]]
    ) -> None:
        if isinstance(pipeline, list):
            for sub_pipeline in pipeline:
                self._set_pipeline_size_value(sub_pipeline, scale)
            return

        updated = False
        for pipeline_name, pipeline_attrs in self.PIPELINE_TO_CHANGE.items():
            if pipeline_name in pipeline["type"].lower():
                for pipeline_attr in pipeline_attrs:
                    if pipeline_attr in pipeline:
                        self._set_size_value(pipeline, pipeline_attr, scale)
                        updated = True
                if updated:
                    break

        for pipeline_name, sub_pipeline_names in self.PIPELINE_WRAPPER.items():
            if pipeline_name == pipeline["type"]:
                for sub_pipeline_name in sub_pipeline_names:
                    if sub_pipeline_name in pipeline:
                        if isinstance(pipeline[sub_pipeline_name], dict):
                            self._set_pipeline_size_value(pipeline[sub_pipeline_name], scale)
                        elif isinstance(pipeline[sub_pipeline_name][0], dict):
                            for sub_pipeline in pipeline[sub_pipeline_name]:
                                self._set_pipeline_size_value(sub_pipeline, scale)
                        elif isinstance(pipeline[sub_pipeline_name][0], list):
                            for sub_pipelines in pipeline[sub_pipeline_name]:
                                for sub_pipeline in sub_pipelines:
                                    self._set_pipeline_size_value(sub_pipeline, scale)
                        else:
                            raise ValueError(
                                "Dataset pipeline in pipeline wrapper type should be"
                                "either dict, list[dict] or list[list[dict]]."
                            )

    @staticmethod
    def _set_size_value(pipeline: Dict, attr: str, scale: Tuple[Union[int, float], Union[int, float]]) -> None:
        if isinstance(pipeline[attr], int):
            pipeline[attr] = round(pipeline[attr] * scale[0])
        elif isinstance(pipeline[attr], list) and isinstance(pipeline[attr][0], tuple):
            for idx in range(len(pipeline[attr])):
                pipeline[attr][idx] = (
                    round(pipeline[attr][idx][0] * scale[0]),
                    round(pipeline[attr][idx][1] * scale[1]),
                )
        else:
            pipeline[attr] = (round(pipeline[attr][0] * scale[0]), round(pipeline[attr][1] * scale[1]))

    @staticmethod
    def get_trained_input_size(model_ckpt: Optional[str] = None) -> Optional[Tuple[int, int]]:
        """Get trained input size from checkpoint. If it doesn't exist, return None.

        Args:
            model_ckpt (Optional[str], optional): Model weight to load. Defaults to None.

        Returns:
            Optional[Tuple[int, int]]: Pair of width and height. If there is no input size configuration, return None.
        """
        if model_ckpt is None:
            return None

        model_info = torch.load(model_ckpt, map_location="cpu")
        if model_info is None:
            return None

        input_size = model_info.get("input_size", None)
        if not input_size:
            return None

        logger.info("Given model weight was trained with {} input size.".format(input_size))
        return input_size

    @staticmethod
    def select_closest_size(input_size: Tuple[int, int], preset_sizes: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Select the most closest size from preset sizes in log scale.

        Args:
            input_size (Tuple[int, int]): Query input size
            preset_sizes (List[Tuple[int, int]]): List of preset input sizes

        Returns:
            Tuple[int, int]: Best matching size out of preset. Returns input_size if preset is empty.
        """
        if len(preset_sizes) == 0:
            return input_size

        def to_log_scale(x: tuple) -> np.ndarray:
            return np.log(np.sqrt(x[0] * x[1]))

        input_scale = to_log_scale(input_size)
        preset_scales = np.array(list(map(to_log_scale, preset_sizes)))
        abs_diff = np.abs(preset_scales - input_scale)
        return preset_sizes[np.argmin(abs_diff)]

    def adapt_input_size_to_dataset(
        self, max_image_size: int, min_object_size: Optional[int] = None, downscale_only: bool = True
    ) -> Union[Tuple[int, int], Dict[str, Tuple[int, int]]]:
        """Compute appropriate model input size w.r.t. dataset statistics.

        Args:
            max_image_size (int): Typical large image size of dataset in pixels.
            min_object_size (int, optional): Typical small object size of dataset in pixels.
                None to consider only image size. Defaults to None.
            downscale_only (bool) : Whether to allow only smaller size than default setting. Defaults to True.

        Returns:
            Tuple[int, int]: (width, height)
        """

        logger.info("Adapting model input size based on dataset stat")

        base_input_size = self.base_input_size
        if isinstance(base_input_size, Dict):
            base_input_size = base_input_size.get("train", base_input_size.get("test", {}))
        logger.info(f"-> Current base input size: {base_input_size}")

        if max_image_size <= 0:
            return base_input_size

        image_size = max_image_size
        logger.info(f"-> Based on typical large image size: {image_size}")

        # Refine using annotation shape size stat
        if min_object_size is not None and min_object_size > 0:
            image_size = round(image_size * self.MIN_RECOGNIZABLE_OBJECT_SIZE / min_object_size)
            logger.info(f"-> Based on typical small object size {min_object_size}: {image_size}")
            if image_size > max_image_size:
                image_size = max_image_size
                logger.info(f"-> Restrict to max image size: {image_size}")
            if image_size < self.MIN_DETECTION_INPUT_SIZE:
                image_size = self.MIN_DETECTION_INPUT_SIZE
                logger.info(f"-> Based on minimum object detection input size: {image_size}")

        input_size = (round(image_size), round(image_size))

        if downscale_only:

            def area(x: Tuple[int, int]) -> int:
                return x[0] * x[1]

            if base_input_size and isinstance(base_input_size, tuple) and area(input_size) >= area(base_input_size):
                logger.info(f"-> Downscale only: {input_size} -> {base_input_size}")
                return base_input_size

        # Closest preset
        input_size_preset = InputSizePreset.input_sizes()
        input_size = self.select_closest_size(input_size, input_size_preset)
        logger.info(f"-> Closest preset: {input_size}")
        return input_size
