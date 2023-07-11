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
import shutil
import sys
import tempfile
import warnings
from collections.abc import Mapping
from importlib import import_module
from typing import Any, Callable, Dict, List, Tuple, Union

import torch
from mmcv import Config, ConfigDict
from mmcv.utils.config import BASE_KEY, DEPRECATION_KEY
from mmcv.utils.misc import import_modules_from_strings
from mmcv.utils.path import check_file_exist

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
    if config.runner.type != "AccuracyAwareRunner":
        if is_epoch_based_runner(config.runner) and config.runner.type != "EpochRunnerWithCancel":
            logger.info(f"Replacing runner from {config.runner.type} to EpochRunnerWithCancel.")
            config.runner.type = "EpochRunnerWithCancel"
        elif not is_epoch_based_runner(config.runner) and config.runner.type != "IterBasedRunnerWithCancel":
            logger.info(f"Replacing runner from {config.runner.type} to IterBasedRunnerWithCancel.")
            config.runner.type = "IterBasedRunnerWithCancel"


def patch_fp16(config: Config):
    """Remove FP16 config if running on CPU device and revert to FP32.

    Please refer https://github.com/pytorch/pytorch/issues/23377
    """
    if not torch.cuda.is_available() and "fp16" in config:
        logger.info("Revert FP16 to FP32 on CPU device")
        if isinstance(config, Config):
            del config._cfg_dict["fp16"]  # pylint: disable=protected-access
        elif isinstance(config, ConfigDict):
            del config["fp16"]


def patch_adaptive_interval_training(config: Config):
    """Update adaptive interval settings for OTX training.

    This function can be removed by adding custom hook cfg into recipe.py directly.
    """
    # default adaptive hook for evaluating before and after training
    add_custom_hook_if_not_exists(
        config,
        ConfigDict(
            type="AdaptiveTrainSchedulingHook",
            enable_adaptive_interval_hook=False,
            enable_eval_before_run=True,
        ),
    )
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
