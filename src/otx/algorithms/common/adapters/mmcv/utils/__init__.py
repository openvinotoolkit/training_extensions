"""OTX Adapters - mmcv.utils."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from ._builder_build_data_parallel import build_data_parallel
from ._config_utils_get_configs_by_keys import get_configs_by_keys
from ._config_utils_get_configs_by_pairs import get_configs_by_pairs
from .automatic_bs import adapt_batch_size
from .builder import build_dataloader, build_dataset
from .config_utils import (
    MPAConfig,
    align_data_config_with_recipe,
    config_from_string,
    get_data_cfg,
    get_dataset_configs,
    get_meta_keys,
    is_epoch_based_runner,
    patch_adaptive_interval_training,
    patch_color_conversion,
    patch_data_pipeline,
    patch_default_config,
    patch_early_stopping,
    patch_fp16,
    patch_from_hyperparams,
    patch_persistent_workers,
    patch_runner,
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
    "patch_default_config",
    "patch_data_pipeline",
    "patch_color_conversion",
    "patch_early_stopping",
    "patch_from_hyperparams",
    "patch_fp16",
    "patch_persistent_workers",
    "patch_runner",
    "align_data_config_with_recipe",
    "get_meta_keys",
    "prepare_work_dir",
    "get_data_cfg",
    "MPAConfig",
    "adapt_batch_size",
]
