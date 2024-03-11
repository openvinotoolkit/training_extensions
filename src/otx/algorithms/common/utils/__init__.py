"""Collection of utils to run common OTX algorithms."""

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

import os

from .callback import (
    InferenceProgressCallback,
    OptimizationProgressCallback,
    TrainingProgressCallback,
)
from .data import OTXOpenVinoDataLoader, get_cls_img_indices, get_image, get_old_new_img_indices
from .dist_utils import append_dist_rank_suffix
from .ir import embed_ir_model_data
from .utils import (
    UncopiableDefaultDict,
    cast_bf16_to_fp32,
    get_arg_spec,
    get_cfg_based_on_device,
    get_default_async_reqs_num,
    get_task_class,
    is_hpu_available,
    is_xpu_available,
    load_template,
    read_py_config,
    set_random_seed,
)

__all__ = [
    "embed_ir_model_data",
    "get_cls_img_indices",
    "get_old_new_img_indices",
    "TrainingProgressCallback",
    "InferenceProgressCallback",
    "OptimizationProgressCallback",
    "UncopiableDefaultDict",
    "load_template",
    "get_task_class",
    "get_arg_spec",
    "get_image",
    "set_random_seed",
    "append_dist_rank_suffix",
    "OTXOpenVinoDataLoader",
    "read_py_config",
    "get_default_async_reqs_num",
    "is_xpu_available",
    "is_hpu_available",
    "cast_bf16_to_fp32",
    "get_cfg_based_on_device",
]


if is_hpu_available():
    os.environ["PT_HPU_LAZY_MODE"] = "1"
    import habana_frameworks.torch.gpu_migration  # noqa: F401


if is_xpu_available():
    try:
        import mmcv

        from otx.algorithms.common.adapters.mmcv.utils.fp16_utils import custom_auto_fp16, custom_force_fp32

        mmcv.runner.auto_fp16 = custom_auto_fp16
        mmcv.runner.force_fp32 = custom_force_fp32
    except ImportError:
        pass
