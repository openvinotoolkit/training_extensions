"""This module implements type related utility functions."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


from otx.v2.api.entities.subset import Subset
from otx.v2.api.entities.task_type import TaskType, TrainType

map_task_type = {str(task_type.name).upper(): task_type for task_type in TaskType}
map_train_type = {str(train_type.name).upper(): train_type for train_type in TrainType}
map_subset_type = {str(subset_type.name).upper(): subset_type for subset_type in Subset}


def str_to_task_type(task_type: str) -> TaskType:
    if task_type.upper() in map_task_type:
        return map_task_type[task_type.upper()]
    msg = f"{task_type.upper()} is not supported task."
    raise ValueError(msg)


def str_to_train_type(train_type: str) -> TrainType:
    if train_type.upper() in map_train_type:
        return map_train_type[train_type.upper()]
    msg = f"{train_type.upper()} is not supported train type."
    raise ValueError(msg)


def str_to_subset_type(subset: str) -> Subset:
    map_short_str = {"train": "training", "val": "validation", "test": "testing", "unlabel": "unlabeled"}
    if subset in map_short_str:
        subset = map_short_str[subset]
    if subset.upper() in map_subset_type:
        return map_subset_type[subset.upper()]
    msg = f"{subset.upper()} is not supported subset type."
    raise ValueError(msg)
