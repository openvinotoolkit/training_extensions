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

from typing import List

from mmcv import Config, ConfigDict

from otx.algorithms.common.adapters.mmcv.utils import (
    get_dataset_configs,
)
from otx.utils.logger import get_logger

logger = get_logger()


def patch_datasets(
    config: Config,
    subsets: List[str] = ["train", "val", "test", "unlabeled"],
    **kwargs,
):
    """Update dataset configs."""
    assert "data" in config
    assert "type" in kwargs

    for subset in subsets:
        if subset not in config.data:
            continue
        config.data[f"{subset}_dataloader"] = config.data.get(f"{subset}_dataloader", ConfigDict())

        # For stable hierarchical information indexing
        if subset == "train" and kwargs["type"] == "OTXHierarchicalClsDataset":
            config.data[f"{subset}_dataloader"]["drop_last"] = True

        cfgs = get_dataset_configs(config, subset)
        for cfg in cfgs:
            cfg.update(kwargs)


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
