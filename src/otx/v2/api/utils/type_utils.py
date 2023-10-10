"""Module implementing type related utility functions."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


from otx.v2.api.entities.subset import Subset
from otx.v2.api.entities.task_type import TaskType, TrainType

map_task_type = {str(task_type.name).upper(): task_type for task_type in TaskType}
map_train_type = {str(train_type.name).upper(): train_type for train_type in TrainType}
map_subset_type = {str(subset_type.name).upper(): subset_type for subset_type in Subset}


def str_to_task_type(task_type: str) -> TaskType:
    """Convert a string to a TaskType enum.

    Args:
    ----
        task_type (str): The string representation of the task type.

    Returns:
    -------
        TaskType: The corresponding TaskType enum.

    Raises:
    ------
        ValueError: If the task type is not supported.
    """
    if task_type.upper() in map_task_type:
        return map_task_type[task_type.upper()]
    msg = f"{task_type.upper()} is not supported task."
    raise ValueError(msg)


def str_to_train_type(train_type: str) -> TrainType:
    """Convert a string train type to a TrainType enum.

    Args:
    ----
        train_type (str): The string representation of the train type.

    Returns:
    -------
        TrainType: The corresponding TrainType enum.

    Raises:
    ------
        ValueError: If the train type is not supported.
    """
    if train_type.upper() in map_train_type:
        return map_train_type[train_type.upper()]
    msg = f"{train_type.upper()} is not supported train type."
    raise ValueError(msg)


def str_to_subset_type(subset: str) -> Subset:
    """Convert a string representation of a subset type to a Subset enum.

    Args:
    ----
        subset (str): The string representation of the subset type.

    Returns:
    -------
        Subset: The corresponding Subset enum.

    Raises:
    ------
        ValueError: If the subset type is not supported.
    """
    map_short_str = {"train": "training", "val": "validation", "test": "testing", "unlabel": "unlabeled"}
    if subset in map_short_str:
        subset = map_short_str[subset]
    if subset.upper() in map_subset_type:
        return map_subset_type[subset.upper()]
    msg = f"{subset.upper()} is not supported subset type."
    raise ValueError(msg)
