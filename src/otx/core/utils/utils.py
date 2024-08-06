# Copyright (c) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions."""

from __future__ import annotations

import importlib
from collections import defaultdict
from multiprocessing import cpu_count
from typing import TYPE_CHECKING, Any

import torch
from datumaro.components.annotation import AnnotationType, LabelCategories

if TYPE_CHECKING:
    from datumaro import Dataset as DmDataset


def is_ckpt_from_otx_v1(ckpt: dict) -> bool:
    """Check the checkpoint where it comes from.

    Args:
        ckpt (dict): the checkpoint file

    Returns:
        bool: True means the checkpoint comes from otx1
    """
    return "model" in ckpt and ckpt["VERSION"] == 1


def is_ckpt_for_finetuning(ckpt: dict) -> bool:
    """Check the checkpoint will be used to finetune.

    Args:
        ckpt (dict): the checkpoint file

    Returns:
        bool: True means the checkpoint will be used to finetune.
    """
    return "state_dict" in ckpt


def get_adaptive_num_workers(num_dataloader: int = 1) -> int | None:
    """Measure appropriate num_workers value and return it."""
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return None
    return min(cpu_count() // (num_dataloader * num_gpus), 8)  # max available num_workers is 8


def get_idx_list_per_classes(dm_dataset: DmDataset, use_string_label: bool = False) -> dict[int | str, list[int]]:
    """Compute class statistics."""
    stats: dict[int | str, list[int]] = defaultdict(list)
    labels = dm_dataset.categories().get(AnnotationType.label, LabelCategories())
    for item_idx, item in enumerate(dm_dataset):
        for ann in item.annotations:
            if use_string_label:
                stats[labels.items[ann.label].name].append(item_idx)
            else:
                stats[ann.label].append(item_idx)
    # Remove duplicates in label stats idx: O(n)
    for k in stats:
        stats[k] = list(dict.fromkeys(stats[k]))
    return stats


def import_object_from_module(obj_path: str) -> Any:  # noqa: ANN401
    """Get object from import format string."""
    module_name, obj_name = obj_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


def remove_state_dict_prefix(state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Remove prefix from state_dict keys."""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace(prefix, "")
        new_state_dict[new_key] = value
    return new_state_dict
