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
import multiprocessing
import os
import os.path as osp
import platform
import re
import shutil
import sys
import tempfile
import warnings
from collections.abc import Mapping
from importlib import import_module
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from mmcv import Config, ConfigDict
from mmcv.utils.config import BASE_KEY, DEPRECATION_KEY
from mmcv.utils.misc import import_modules_from_strings
from mmcv.utils.path import check_file_exist

from otx.algorithms.common.configs.configuration_enums import InputSizePreset
from otx.algorithms.common.utils.logger import get_logger
from otx.api.entities.datasets import DatasetEntity

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
        return MPAConfig(cfg_dict, cfg_text=cfg_text, filename=filename)

    @property
    def pretty_text(self):
        """Make python file human-readable.

        It's almost same as mmcv.Config's code but code to reformat using yapf is removed to reduce time.
        """

        indent = 4

        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        def _format_basic_types(k, v, use_mapping=False):
            if isinstance(v, str):
                v_str = f"'{v}'"
            else:
                v_str = str(v)

            if use_mapping:
                k_str = f"'{k}'" if isinstance(k, str) else str(k)
                attr_str = f"{k_str}: {v_str}"
            else:
                attr_str = f"{str(k)}={v_str}"
            attr_str = _indent(attr_str, indent)

            return attr_str

        def _format_list(k, v, use_mapping=False):
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

        def _contain_invalid_identifier(dict_str):
            contain_invalid_identifier = False
            for key_name in dict_str:
                contain_invalid_identifier |= not str(key_name).isidentifier()
            return contain_invalid_identifier

        def _format_dict(input_dict, outest_level=False):
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


def copy_config(cfg):
    """A function that creates a deep copy of the input configuration object.

    :param cfg: Config object, an instance of `Config` class to be copied.
    :return: Config object, a deep copy of the input configuration object.
    :raises: ValueError if the input object is not an instance of `Config` class.
    """
    if not isinstance(cfg, Config):
        raise ValueError(f"cannot copy this instance {type(cfg)}")

    # disable [B301, B403] pickle, import-pickle - the library used for converting cfg object
    import pickle  # nosec B403

    data = pickle.dumps(cfg)
    return pickle.loads(data)  # nosec B301


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


def remove_from_config(config: Union[Config, ConfigDict], key: str):
    """Update & Remove configs."""
    if key in config:
        if isinstance(config, Config):
            del config._cfg_dict[key]  # pylint: disable=protected-access
        elif isinstance(config, ConfigDict):
            del config[key]
        else:
            raise ValueError(f"Unknown config type {type(config)}")


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


def prepare_for_testing(config: Union[Config, ConfigDict], dataset: DatasetEntity) -> Config:
    """Prepare configs for testing phase."""
    config = copy.deepcopy(config)
    # FIXME. Should working directories be modified here?
    config.data.test.otx_dataset = dataset
    return config


def is_epoch_based_runner(runner_config: ConfigDict):
    """Check Epoch based or Iter based runner."""
    return "Epoch" in runner_config.type


def config_from_string(config_string: str) -> Config:
    """Generate an mmcv config dict object from a string.

    :param config_string: string to parse
    :return config: configuration object
    """
    with tempfile.NamedTemporaryFile("w", suffix=".py") as temp_file:
        temp_file.write(config_string)
        temp_file.flush()
        return Config.fromfile(temp_file.name)


def patch_data_pipeline(config: Config, data_pipeline: str = ""):
    """Replace data pipeline to data_pipeline.py if it exist."""
    if os.path.isfile(data_pipeline):
        data_pipeline_cfg = Config.fromfile(data_pipeline)
        config.merge_from_dict(data_pipeline_cfg)
    else:
        raise FileNotFoundError(f"data_pipeline: {data_pipeline} not founded")


def patch_color_conversion(config: Config):
    """Patch color conversion."""
    assert "data" in config

    for cfg in get_configs_by_pairs(config.data, dict(type="Normalize")):
        to_rgb = False
        if "to_rgb" in cfg:
            to_rgb = cfg.to_rgb
        cfg.to_rgb = not bool(to_rgb)


def patch_adaptive_interval_training(config: Config):
    """Update adaptive interval settings for OTX training.

    This function can be removed by adding custom hook cfg into recipe.py directly.
    """
    # Add/remove adaptive interval hook
    if config.get("use_adaptive_interval", False):
        update_or_add_custom_hook(
            config,
            ConfigDict(
                {
                    "type": "AdaptiveTrainSchedulingHook",
                    "max_interval": 5,
                    "enable_adaptive_interval_hook": True,
                    "enable_eval_before_run": True,
                    **config.pop("adaptive_validation_interval", {}),
                }
            ),
        )
    else:
        config.pop("adaptive_validation_interval", None)


def patch_early_stopping(config: Config):
    """Update early stop settings for OTX training.

    This function can be removed by adding custom hook cfg into recipe.py directly.
    """
    if "early_stop" in config:
        remove_custom_hook(config, "EarlyStoppingHook")
        early_stop = config.get("early_stop", False)
        if early_stop:
            early_stop_hook = ConfigDict(
                type="LazyEarlyStoppingHook",
                start=early_stop.start,
                patience=early_stop.patience,
                iteration_patience=early_stop.iteration_patience,
                interval=1,
                metric=config.early_stop_metric,
                priority=75,
            )
            update_or_add_custom_hook(config, early_stop_hook)
        else:
            remove_custom_hook(config, "LazyEarlyStoppingHook")

    # make sure model to be in a training mode even after model is evaluated (mmcv bug)
    update_or_add_custom_hook(
        config,
        ConfigDict(type="ForceTrainModeHook", priority="LOWEST"),
    )


def patch_persistent_workers(config: Config):
    """If num_workers is 0, persistent_workers must be False."""
    data_cfg = config.data
    for subset in ["train", "val", "test", "unlabeled"]:
        if subset not in data_cfg:
            continue
        dataloader_cfg = data_cfg.get(f"{subset}_dataloader", ConfigDict())
        workers_per_gpu = dataloader_cfg.get(
            "workers_per_gpu",
            data_cfg.get("workers_per_gpu", 0),
        )
        if workers_per_gpu == 0:
            dataloader_cfg["persistent_workers"] = False
        elif "persistent_workers" not in dataloader_cfg:
            dataloader_cfg["persistent_workers"] = True

        if "pin_memory" not in dataloader_cfg:
            dataloader_cfg["pin_memory"] = True

        data_cfg[f"{subset}_dataloader"] = dataloader_cfg


def get_adaptive_num_workers() -> Union[int, None]:
    """Measure appropriate num_workers value and return it."""
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        logger.warning("There is no GPUs. Use existing num_worker value.")
        return None
    return min(multiprocessing.cpu_count() // num_gpus, 8)  # max available num_workers is 8


def patch_from_hyperparams(config: Config, hyperparams):
    """Patch config parameters from hyperparams."""
    params = hyperparams.learning_parameters
    warmup_iters = int(params.learning_rate_warmup_iters)

    model_label_type = config.filename.split("/")[-1]
    if "multilabel" in model_label_type:
        lr_config = ConfigDict(max_lr=params.learning_rate, warmup=None)
    else:
        lr_config = (
            ConfigDict(warmup_iters=warmup_iters)
            if warmup_iters > 0
            else ConfigDict(warmup_iters=warmup_iters, warmup=None)
        )

    if params.enable_early_stopping and config.get("evaluation", None):
        early_stop = ConfigDict(
            start=int(params.early_stop_start),
            patience=int(params.early_stop_patience),
            iteration_patience=int(params.early_stop_iteration_patience),
        )
    else:
        early_stop = False

    runner = ConfigDict(max_epochs=int(params.num_iters))
    if config.get("runner", None) and config.runner.get("type").startswith("IterBasedRunner"):
        runner = ConfigDict(max_iters=int(params.num_iters))

    hparams = ConfigDict(
        optimizer=ConfigDict(lr=params.learning_rate),
        lr_config=lr_config,
        early_stop=early_stop,
        data=ConfigDict(
            samples_per_gpu=int(params.batch_size),
            workers_per_gpu=int(params.num_workers),
        ),
        runner=runner,
    )

    if hyperparams.learning_parameters.auto_num_workers:
        adapted_num_worker = get_adaptive_num_workers()
        if adapted_num_worker is not None:
            hparams.data.workers_per_gpu = adapted_num_worker

    if hyperparams.algo_backend.train_type.name == "Semisupervised":
        unlabeled_config = ConfigDict(
            data=ConfigDict(
                unlabeled_dataloader=ConfigDict(
                    samples_per_gpu=int(params.unlabeled_batch_size),
                    workers_per_gpu=int(params.num_workers),
                )
            )
        )
        config.update(unlabeled_config)

    hparams["use_adaptive_interval"] = hyperparams.learning_parameters.use_adaptive_interval
    config.merge_from_dict(hparams)


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


def get_meta_keys(pipeline_step, add_meta_keys: List[str] = []):
    """Update meta_keys for ignore_labels."""
    meta_keys = list(pipeline_step.get("meta_keys", DEFAULT_META_KEYS))
    meta_keys.append("ignored_labels")
    meta_keys += add_meta_keys
    pipeline_step["meta_keys"] = set(meta_keys)
    return pipeline_step


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


def get_data_cfg(config: Union[Config, ConfigDict], subset: str = "train") -> Config:
    """Return dataset configs."""
    data_cfg = config.data[subset]
    while "dataset" in data_cfg:
        data_cfg = data_cfg.dataset
    return data_cfg


class InputSizeManager:
    """Class for changing input size and getting input size value by checking data pipeline.

    NOTE: "resize", "pad", "crop", "mosaic", "randomaffine", "multiscaleflipaug" , "AutoAugment" and "TwoCropTransform"
    are considered at now. If other data pipelines exist, it can work differently than expected.

    Args:
        data_config (Dict): Data configuration expected to have "train", "val" or "test" data pipeline.
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
    }
    SUBSET_TYPES: Tuple[str, str, str, str] = ("train", "val", "test", "unlabeled")

    def __init__(
        self,
        data_config: Dict,
        base_input_size: Optional[Union[int, Tuple[int, int], Dict[str, int], Dict[str, Tuple[int, int]]]] = None,
    ):
        self._data_config = data_config
        if isinstance(base_input_size, int):
            base_input_size = (base_input_size, base_input_size)
        elif isinstance(base_input_size, dict):
            for task in base_input_size.keys():
                if isinstance(base_input_size[task], int):
                    base_input_size[task] = (base_input_size[task], base_input_size[task])  # type: ignore[assignment]
            for subset_type in self.SUBSET_TYPES:
                if subset_type in data_config and subset_type not in base_input_size:
                    raise ValueError(
                        f"There is {subset_type} data configuration but base input size for it doesn't exists."
                    )

        self._base_input_size = base_input_size

    def set_input_size(self, input_size: Union[int, List[int], Tuple[int, int]]):
        """Set input size in data pipe line.

        Args:
            input_size (Union[int, List[int]]):
                input size to set. If it's an integer, (input_size x input_size) will be set.
                If input_size is an integer list, (input_size[0] x input_size[1]) will be set.
        """
        if isinstance(input_size, int):
            input_size = [input_size, input_size]
        if not isinstance(self.base_input_size, dict):
            resize_ratio = (input_size[0] / self.base_input_size[0], input_size[1] / self.base_input_size[1])

        # scale size values
        for subset_type in self.SUBSET_TYPES:
            if subset_type in self._data_config:
                if isinstance(self.base_input_size, dict):
                    resize_ratio = (
                        input_size[0] / self.base_input_size[subset_type][0],
                        input_size[1] / self.base_input_size[subset_type][1],
                    )
                pipelines = self._get_pipelines(subset_type)
                for pipeline in pipelines:
                    self._set_pipeline_size_value(pipeline, resize_ratio)

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
    ) -> Union[None, Tuple[int, int]]:
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
        self, pipelines: List[Dict], default_size: Optional[List[int]] = None
    ) -> Union[List[int], None]:
        # NOTE: Mosaic isn't considered in this step because Mosaic and following RandomAffine don't change image size
        post_img_size = default_size
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
                        if isinstance(sub_pipeline[0], list):
                            sub_pipeline = sub_pipeline[0]
                        post_img_size = self._estimate_post_img_size(sub_pipeline, post_img_size)
                        break

            if pipeline["type"] == "LoadResizeDataFromOTXDataset":  # Separate condition due to 'resize' in type name
                if "resize_cfg" in pipeline:
                    post_img_size = self._estimate_post_img_size([pipeline["resize_cfg"]], post_img_size)

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

    def _get_pipelines(self, subset_type: str):
        if "pipeline" in self._data_config[subset_type]:
            return self._data_config[subset_type]["pipeline"]
        if "dataset" in self._data_config[subset_type]:
            return self._data_config[subset_type]["dataset"]["pipeline"]
        raise RuntimeError("Failed to find pipeline.")

    def _set_pipeline_size_value(self, pipeline: Dict, scale: Tuple[Union[int, float], Union[int, float]]):
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
                        if isinstance(pipeline[sub_pipeline_name][0], dict):
                            for sub_pipeline in pipeline[sub_pipeline_name]:
                                self._set_pipeline_size_value(sub_pipeline, scale)
                        elif isinstance(pipeline[sub_pipeline_name][0], list):
                            for sub_pipelines in pipeline[sub_pipeline_name]:
                                for sub_pipeline in sub_pipelines:
                                    self._set_pipeline_size_value(sub_pipeline, scale)
                        else:
                            raise ValueError(
                                "Dataset pipeline in pipeline wrapper type should be"
                                "either list[dict] or list[list[dict]]."
                            )

    @staticmethod
    def _set_size_value(pipeline: Dict, attr: str, scale: Tuple[Union[int, float], Union[int, float]]):
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


def get_configured_input_size(
    input_size_config: InputSizePreset = InputSizePreset.DEFAULT, model_ckpt: Optional[str] = None
) -> Union[None, Tuple[int, int]]:
    """Get configurable input size configuration. If it doesn't exist, return None.

    Args:
        input_size_config (InputSizePreset, optional): Input size configuration. Defaults to InputSizePreset.DEFAULT.
        model_ckpt (Optional[str], optional): Model weight to load. Defaults to None.

    Returns:
        Union[None, Tuple[int, int]]: Pair of width and height. If there is no input size configuration, return None.
    """
    input_size = None
    if input_size_config == InputSizePreset.DEFAULT:
        if model_ckpt is None:
            return None

        model_info = torch.load(model_ckpt, map_location="cpu")
        for key in ["config", "learning_parameters", "input_size", "value"]:
            if key not in model_info:
                return None
            model_info = model_info[key]
        input_size = model_info

        if input_size == InputSizePreset.DEFAULT.value:
            return None

        logger.info("Given model weight was trained with {} input size.".format(input_size))
    else:
        input_size = input_size_config.value

    parsed_tocken = re.match("(\\d+)x(\\d+)", input_size)
    return (int(parsed_tocken.group(1)), int(parsed_tocken.group(2)))
