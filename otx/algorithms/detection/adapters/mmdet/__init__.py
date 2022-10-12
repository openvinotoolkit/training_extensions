"""MMdetection Adapters."""

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

from .config_utils import (
    cluster_anchors,
    config_from_string,
    patch_config,
    prepare_for_testing,
    prepare_for_training,
    set_hyperparams,
)
from .dataset import MPADetDataset
from .pipelines import LoadAnnotationFromOTXDataset, LoadImageFromOTXDataset

__all__ = [
    "MPADetDataset",
    "cluster_anchors",
    "config_from_string",
    "patch_config",
    "prepare_for_testing",
    "prepare_for_training",
    "set_hyperparams",
    "LoadAnnotationFromOTXDataset",
    "LoadImageFromOTXDataset",
]
