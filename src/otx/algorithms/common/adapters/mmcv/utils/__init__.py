"""OTX Adapters - mmcv.utils."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from ._builder_build_data_parallel import HPUDataParallel, XPUDataParallel, build_data_parallel
from ._config_utils_get_configs_by_keys import get_configs_by_keys
from ._config_utils_get_configs_by_pairs import get_configs_by_pairs
from .automatic_bs import adapt_batch_size
from .builder import build_dataloader, build_dataset
from .config_utils import (
    InputSizeManager,
    OTXConfig,
    config_from_string,
    get_dataset_configs,
    is_epoch_based_runner,
    patch_adaptive_interval_training,
    patch_color_conversion,
    patch_early_stopping,
    patch_from_hyperparams,
    patch_persistent_workers,
    prepare_for_testing,
    prepare_work_dir,
    remove_from_config,
    remove_from_configs_by_type,
    update_config,
)

__all__ = [
    "build_dataset",
    "build_dataloader",
    "build_data_parallel",
    "remove_from_config",
    "remove_from_configs_by_type",
    "get_configs_by_pairs",
    "get_configs_by_keys",
    "update_config",
    "get_dataset_configs",
    "prepare_for_testing",
    "is_epoch_based_runner",
    "config_from_string",
    "patch_adaptive_interval_training",
    "patch_color_conversion",
    "patch_early_stopping",
    "patch_persistent_workers",
    "prepare_work_dir",
    "OTXConfig",
    "adapt_batch_size",
    "InputSizeManager",
    "XPUDataParallel",
    "HPUDataParallel",
    "patch_from_hyperparams",
]
