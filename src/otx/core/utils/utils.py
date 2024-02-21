# Copyright (c) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions."""

from __future__ import annotations

from multiprocessing import cpu_count
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from omegaconf import DictConfig


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


def get_mean_std_from_data_processing(config: DictConfig) -> dict[str, Any]:
    """Get mean and std value from data_processing.

    Args:
        config (DictConfig): MM framework model config.

    Returns:
        dict[str, Any]: Dictionary with mean and std value.
    """
    return {
        "mean": config["data_preprocessor"]["mean"],
        "std": config["data_preprocessor"]["std"],
    }


def get_adaptive_num_workers(num_dataloader: int = 1) -> int | None:
    """Measure appropriate num_workers value and return it."""
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return None
    return min(cpu_count() // (num_dataloader * num_gpus), 8)  # max available num_workers is 8
