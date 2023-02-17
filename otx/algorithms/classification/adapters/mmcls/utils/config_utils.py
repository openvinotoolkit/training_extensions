"""Collection of utils for task implementation in Classification Task."""

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

import math
from typing import List, Optional, Union

from mmcv import Config, ConfigDict

from otx.algorithms.common.adapters.mmcv.utils import (
    get_configs_by_keys,
    get_configs_by_pairs,
    get_dataset_configs,
    get_meta_keys,
    is_epoch_based_runner,
    patch_color_conversion,
    prepare_work_dir,
    remove_from_config,
    remove_from_configs_by_type,
    update_config,
)
from otx.api.entities.label import Domain, LabelEntity
from otx.api.utils.argument_checks import (
    DirectoryPathCheck,
    check_input_parameters_type,
)
from otx.mpa.utils.logger import get_logger

logger = get_logger()


@check_input_parameters_type({"work_dir": DirectoryPathCheck})
def patch_config(
    config: Config,
    work_dir: str,
    labels: List[LabelEntity],
):  # pylint: disable=too-many-branches
    """Update config function."""

    # Add training cancelation hook.
    if "custom_hooks" not in config:
        config.custom_hooks = []
    if "CancelTrainingHook" not in {hook.type for hook in config.custom_hooks}:
        config.custom_hooks.append(ConfigDict({"type": "CancelTrainingHook"}))

    # Remove high level data pipelines definition leaving them only inside `data` section.
    remove_from_config(config, "train_pipeline")
    remove_from_config(config, "test_pipeline")
    remove_from_config(config, "train_pipeline_strong")
    # Remove cancel interface hook
    remove_from_configs_by_type(config.custom_hooks, "CancelInterfaceHook")

    config.checkpoint_config.max_keep_ckpts = 5
    config.checkpoint_config.interval = config.evaluation.get("interval", 1)

    set_data_classes(config, labels)

    config.gpu_ids = range(1)
    config.work_dir = work_dir


@check_input_parameters_type()
def patch_model_config(
    config: Config,
    labels: List[LabelEntity],
):
    """Patch model config."""
    set_num_classes(config, len(labels))


@check_input_parameters_type()
def patch_adaptive_repeat_dataset(
    config: Union[Config, ConfigDict],
    num_samples: int,
    decay: float = -0.002,
    factor: float = 30,
):
    """Patch the repeat times and training epochs adatively.

    Frequent dataloading inits and evaluation slow down training when the
    sample size is small. Adjusting epoch and dataset repetition based on
    empirical exponential decay improves the training time by applying high
    repeat value to small sample size dataset and low repeat value to large
    sample.

    :param config: mmcv config
    :param num_samples: number of training samples
    :param decay: decaying rate
    :param factor: base repeat factor
    """
    data_train = config.data.train
    if data_train.type == "RepeatDataset" and getattr(data_train, "adaptive_repeat_times", False):
        if is_epoch_based_runner(config.runner):
            cur_epoch = config.runner.max_epochs
            new_repeat = max(round(math.exp(decay * num_samples) * factor), 1)
            new_epoch = math.ceil(cur_epoch / new_repeat)
            if new_epoch == 1:
                return
            config.runner.max_epochs = new_epoch
            data_train.times = new_repeat


@check_input_parameters_type()
def prepare_for_training(
    config: Union[Config, ConfigDict],
    data_config: ConfigDict,
) -> Union[Config, ConfigDict]:
    """Prepare configs for training phase."""
    prepare_work_dir(config)

    train_num_samples = 0
    for subset in ["train", "val", "test"]:
        data_config_ = data_config.data.get(subset)
        config_ = config.data.get(subset)
        if data_config_ is None:
            continue
        for key in ["otx_dataset", "labels"]:
            found = get_configs_by_keys(data_config_, key, return_path=True)
            if len(found) == 0:
                continue
            assert len(found) == 1
            if subset == "train" and key == "otx_dataset":
                found_value = list(found.values())[0]
                if found_value:
                    train_num_samples = len(found_value)
            update_config(config_, found)

    if train_num_samples > 0:
        patch_adaptive_repeat_dataset(config, train_num_samples)

    return config


@check_input_parameters_type()
def set_data_classes(config: Config, labels: List[LabelEntity]):
    """Setter data classes into config."""
    # Save labels in data configs.
    for subset in ("train", "val", "test"):
        for cfg in get_dataset_configs(config, subset):
            cfg.labels = labels


@check_input_parameters_type()
def set_num_classes(config: Config, num_classes: int):
    """Set num classes."""
    head_names = ["head"]
    for head_name in head_names:
        if head_name in config.model:
            config.model[head_name].num_classes = num_classes


@check_input_parameters_type()
def patch_datasets(
    config: Config,
    domain: Domain = Domain.CLASSIFICATION,
    subsets: Optional[List[str]] = None,
    **kwargs,
):
    """Update dataset configs."""
    assert "data" in config
    assert "type" in kwargs

    if subsets is None:
        subsets = ["train", "val", "test", "unlabeled"]

    def update_pipeline(cfg):
        if subset == "train":
            for collect_cfg in get_configs_by_pairs(cfg, dict(type="Collect")):
                get_meta_keys(collect_cfg)

    for subset in subsets:
        if subset not in config.data:
            continue
        config.data[f"{subset}_dataloader"] = config.data.get(f"{subset}_dataloader", ConfigDict())

        # For stable hierarchical information indexing
        if subset == "train" and kwargs["type"] == "OTXHierarchicalClsDataset":
            config.data[f"{subset}_dataloader"].drop_last = True

        cfgs = get_dataset_configs(config, subset)
        for cfg in cfgs:
            cfg.domain = domain
            cfg.otx_dataset = None
            cfg.labels = None
            cfg.update(kwargs)

            update_pipeline(cfg)

    patch_color_conversion(config)


def patch_evaluation(config: Config, task: str):
    """Patch evaluation."""
    cfg = config.get("evaluation", None)
    if cfg:
        if task == "multilabel":
            cfg.metric = ["accuracy-mlc", "mAP", "CP", "OP", "CR", "OR", "CF1", "OF1"]
            config.early_stop_metric = "mAP"
        elif task == "hierarchical":
            cfg.metric = ["MHAcc", "avgClsAcc", "mAP"]
            config.early_stop_metric = "MHAcc"
        elif task == "normal":
            cfg.metric = ["accuracy", "class_accuracy"]
            config.early_stop_metric = "accuracy"
        else:
            raise NotImplementedError
