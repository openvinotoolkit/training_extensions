"""OTX Adapters - deep_object_reid.scripts."""

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

from .config import (
    get_default_config,
    imagedata_kwargs,
    lr_scheduler_kwargs,
    merge_from_files_with_base,
    model_kwargs,
    optimizer_kwargs,
)

__all__ = [
    "get_default_config",
    "imagedata_kwargs",
    "lr_scheduler_kwargs",
    "merge_from_files_with_base",
    "model_kwargs",
    "optimizer_kwargs",
]
