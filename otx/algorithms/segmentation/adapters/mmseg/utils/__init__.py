"""OTX Adapters - mmseg.utils."""

# Copyright (C) 2023 Intel Corporation
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

from .builder import build_scalar_scheduler, build_segmentor
from .config_utils import (
    patch_config,
    patch_datasets,
    patch_evaluation,
    prepare_for_training,
    set_hyperparams,
)
from .data_utils import get_valid_label_mask_per_batch, load_dataset_items

__all__ = [
    "patch_config",
    "patch_datasets",
    "patch_evaluation",
    "prepare_for_training",
    "set_hyperparams",
    "load_dataset_items",
    "build_scalar_scheduler",
    "build_segmentor",
    "get_valid_label_mask_per_batch",
]
