"""Utils for dynamically importing stuff."""

# Copyright (C) 2021 Intel Corporation
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


import importlib
import inspect

# pylint: disable=protected-access


def get_impl_class(impl_path):
    """Returns a class by its path in package."""

    task_impl_module_name, task_impl_class_name = impl_path.rsplit(".", 1)
    task_impl_module = importlib.import_module(task_impl_module_name)
    task_impl_class = getattr(task_impl_module, task_impl_class_name)

    return task_impl_class


def get_backbone_list(backend):
    """Gather backbone list from backends."""
    if backend in ("otx", "custom"):
        otx_backbones = importlib.import_module("otx.algorithms.common.adapters.mmcv.models.backbones")
        return otx_backbones.__all__
    if backend in ("mmcls", "mmdet", "mmseg"):
        if importlib.util.find_spec(backend):
            mm_backbones = importlib.import_module(f"{backend}.models.backbones")
            return mm_backbones.__all__
    if backend == "pytorchcv":
        if importlib.util.find_spec(backend):
            pytorchcv_backbones = importlib.import_module(f"{backend}.model_provider")
            return pytorchcv_backbones._models
    if backend == "torchvision":
        torchvision_backbone_module = "otx.algorithms.common.adapters.mmcv.models.backbones.torchvision_backbones"
        torchvision_backbones = importlib.import_module(torchvision_backbone_module)
        return [f"{backbone}" for backbone in torchvision_backbones.TORCHVISION_MODELS.keys()]
    raise ValueError(f"{backend} cannot be imported.")


def get_backbone_registry(backends=None):
    """Gather backbone list from backends."""

    custom_imports = []
    # Get OTX Custom + Torchvision registry
    otx_backend_path = "otx.algorithms.common.adapters.mmcv.models"
    otx_backbones = importlib.import_module(otx_backend_path)
    otx_registry = otx_backbones.BACKBONES
    custom_imports.append(otx_backend_path)

    if backends is None:
        backend_list = ["mmcls", "mmdet", "mmseg"]
    elif backends in ("otx", "torchvision"):
        backend_list = []
    elif backends in ("mmdet", "pytorchcv"):
        backend_list = ["mmdet"]
    else:
        backend_list = [backends]

    for backend in backend_list:
        if importlib.util.find_spec(backend):
            mm_backbones = importlib.import_module(f"{backend}.models")
            mm_registry = mm_backbones.BACKBONES
            otx_registry._add_children(mm_registry)
            custom_imports.append(f"{backend}.models")
    return otx_registry, custom_imports


def get_required_args(module):
    """Gather backbone's Required Args."""
    if module is None:
        return []
    required_args = []
    args_signature = inspect.signature(module)
    for arg_key, arg_value in args_signature.parameters.items():
        if arg_value.default is inspect.Parameter.empty:
            required_args.append(arg_key)
            continue
    # Get args from parents
    parent_module = module.__bases__
    while len(parent_module):
        parent_args_signature = inspect.signature(parent_module[0])
        for arg_key, arg_value in parent_args_signature.parameters.items():
            if arg_key == "depth" and "arch" in required_args:
                continue
            if arg_value.default is inspect.Parameter.empty and arg_key not in required_args:
                required_args.append(arg_key)
                continue
        parent_module = parent_module[0].__bases__
    required_args = [arg for arg in required_args if arg not in ("args", "kwargs", "self")]
    return required_args
