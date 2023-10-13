"""Algorithm to limits the training time by limiting the max epochs of the training."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import random

from mmcv.utils import Config

from otx.algorithms.common.adapters.mmcv.utils import (
    remove_from_configs_by_type,
)
from otx.algorithms.common.utils.logger import get_logger
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.model_template import ModelTemplate

logger = get_logger()

SIMPLE_INCREMENTAL_CONFIG: dict = {
    "classification": {
        "deit-tiny": {
            "iter_time": 0.08,
        },
        "efficientnet-b0": {
            "iter_time": 0.1,
        },
        "efficientnet-v2-s": {
            "iter_time": 0.15,
        },
        "mobilenet-v3-large-1x": {
            "iter_time": 0.08,
        },
    },
    "max_time": 300,
    "max_num_val_data": 128,
}


def sample_dataset(dataset, num_samples):
    """Sampling the dataset by randomly."""
    sampling_indices = random.sample(range(0, len(dataset)), num_samples)
    sampled_dataset_candidates = DatasetEntity(items=dataset[sampling_indices])

    return sampled_dataset_candidates


def get_task_from_template(template):
    """Get the task type from the template."""
    return str(template.task_type).lower()


def get_model_name_from_template(template):
    """Get the model name from the template."""
    return str(template.name).lower()


def enable_simple_incr_learning(cfg: Config, template: ModelTemplate):
    """Enable the simple incremental learning.

    Args:
        cfg (Config) : the mmcv format configuration file
        template (ModelTemplate): the model template used for the task
    """
    logger.info("Simple incremental model is enabled.")
    logger.info("- If there is AdaptiveRepeatDataHook, it will be disabled.")
    remove_from_configs_by_type(cfg.custom_hooks, "AdaptiveRepeatDataHook")
    batch_size = cfg.data.train_dataloader.samples_per_gpu

    train_dataset = cfg.data.train.otx_dataset
    len_train_dataset = len(train_dataset)
    current_iters_per_epoch = len_train_dataset // batch_size

    task = get_task_from_template(template)
    model_name = get_model_name_from_template(template)

    model_preset = SIMPLE_INCREMENTAL_CONFIG[task][model_name]
    preset_iter_time = model_preset["iter_time"]

    epoch_time = current_iters_per_epoch * preset_iter_time
    max_epochs = int(SIMPLE_INCREMENTAL_CONFIG["max_time"] / epoch_time)

    logger.info("- Max epochs will be changed, " f"{cfg.runner.max_epochs} -> {max_epochs}")
    cfg.runner.max_epochs = max_epochs

    # Sampling the validation dataset to reduce the validation time
    val_dataset = cfg.data.val.otx_dataset
    sampled_val_dataset = sample_dataset(val_dataset, SIMPLE_INCREMENTAL_CONFIG["max_num_val_data"])
    cfg.data.val.otx_dataset = sampled_val_dataset
    logger.info(f"- sampled {len(sampled_val_dataset)} validation dataset to reduce validation time")
