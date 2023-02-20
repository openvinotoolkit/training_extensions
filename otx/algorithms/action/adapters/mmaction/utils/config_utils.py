"""Collection of utils for task implementation in Action Task."""

# Copyright (C) 2021 Intel Corporation
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

from collections import defaultdict
from typing import List, Union

from mmcv.utils import Config, ConfigDict

from otx.algorithms.common.adapters.mmcv.utils import (
    get_data_cfg,
    patch_data_pipeline,
    prepare_work_dir,
)
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label import LabelEntity
from otx.api.entities.model_template import TaskType
from otx.api.usecases.reporting.time_monitor_callback import TimeMonitorCallback
from otx.api.utils.argument_checks import (
    DatasetParamTypeCheck,
    check_input_parameters_type,
)


@check_input_parameters_type()
def patch_config(config: Config, data_pipeline_path: str, work_dir: str, task_type: TaskType):
    """Patch recipe config suitable to mmaction."""
    # FIXME omnisource is hard coded
    config.omnisource = None
    config.work_dir = work_dir
    patch_data_pipeline(config, data_pipeline_path)
    if task_type == TaskType.ACTION_CLASSIFICATION:
        _patch_cls_datasets(config)
    elif task_type == TaskType.ACTION_DETECTION:
        _patch_det_dataset(config)
    else:
        raise NotImplementedError(f"{task_type} is not supported in action task")


@check_input_parameters_type()
def _patch_cls_datasets(config: Config):
    """Patch cls dataset config suitable to mmaction."""

    assert "data" in config
    for subset in ("train", "val", "test", "unlabeled"):
        cfg = config.data.get(subset, None)
        if not cfg:
            continue
        cfg.type = "OTXActionClsDataset"
        cfg.otx_dataset = None
        cfg.labels = None


@check_input_parameters_type()
def _patch_det_dataset(config: Config):
    """Patch det dataset config suitable to mmaction."""
    assert "data" in config
    for subset in ("train", "val", "test", "unlabeled"):
        cfg = config.data.get(subset, None)
        if not cfg:
            continue
        cfg.type = "OTXActionDetDataset"


@check_input_parameters_type()
def set_data_classes(config: Config, labels: List[LabelEntity], task_type: TaskType):
    """Setter data classes into config."""
    for subset in ("train", "val", "test"):
        cfg = get_data_cfg(config, subset)
        cfg.labels = labels

    # FIXME classification head name is hard-coded
    if task_type == TaskType.ACTION_CLASSIFICATION:
        config.model["cls_head"].num_classes = len(labels)
    elif task_type == TaskType.ACTION_DETECTION:
        config.model["roi_head"]["bbox_head"].num_classes = len(labels) + 1
        if len(labels) < 5:
            config.model["roi_head"]["bbox_head"]["topk"] = len(labels) - 1


@check_input_parameters_type({"train_dataset": DatasetParamTypeCheck, "val_dataset": DatasetParamTypeCheck})
def prepare_for_training(
    config: Union[Config, ConfigDict],
    train_dataset: DatasetEntity,
    val_dataset: DatasetEntity,
    time_monitor: TimeMonitorCallback,
    learning_curves: defaultdict,
) -> Config:
    """Prepare configs for training phase."""
    prepare_work_dir(config)
    data_train = get_data_cfg(config)
    data_train.otx_dataset = train_dataset
    config.data.val.otx_dataset = val_dataset
    config.custom_hooks.append({"type": "OTXProgressHook", "time_monitor": time_monitor, "verbose": True})
    config.log_config.hooks.append({"type": "OTXLoggerHook", "curves": learning_curves})

    return config
