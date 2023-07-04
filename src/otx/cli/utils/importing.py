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
import json
import os

# pylint: disable=protected-access

SUPPORTED_BACKBONE_BACKENDS = {
    "otx": "otx.algorithms.common.adapters.mmcv.models",
    "mmcls": "mmcls.models",
    "mmdet": "mmdet.models",
    "mmseg": "mmseg.models",
    "torchvision": "otx.algorithms.common.adapters.mmcv.models",
    "pytorchcv": "mmdet.models",
    "omz.mmcls": "otx.algorithms.classification.adapters.mmcls.models.backbones.mmov_backbone",
}


def get_impl_class(impl_path):
    """Returns a class by its path in package."""

    task_impl_module_name, task_impl_class_name = impl_path.rsplit(".", 1)
    task_impl_module = importlib.import_module(task_impl_module_name)
    task_impl_class = getattr(task_impl_module, task_impl_class_name)

    return task_impl_class


def get_backbone_list(backend):
    """Gather available backbone list from json file & imported lib."""
    available_backbone_path = os.path.join(get_otx_root_path(), f"cli/builder/supported_backbone/{backend}.json")
    available_backbones = {}
    if os.path.exists(available_backbone_path):
        with open(available_backbone_path, "r", encoding="UTF-8") as f:
            available_backbones = json.load(f)
        available_backbones = available_backbones["backbones"]
    elif backend == "pytorchcv" and importlib.util.find_spec(backend):
        backbone_list = importlib.import_module(f"{backend}.model_provider")._models
        backbone_format = {"required": [], "options": {}, "available": []}
        for backbone in backbone_list:
            backbone_type = f"mmdet.{backbone}"
            available_backbones[backbone_type] = backbone_format
    else:
        raise ValueError(f"{backend} cannot be imported or supported.")
    return available_backbones


def get_backbone_registry(backend=None):
    """Gather backbone list from backends."""
    if backend not in SUPPORTED_BACKBONE_BACKENDS:
        raise ValueError(f"{backend} is an unsupported backbone backend.")

    custom_imports = []
    backend_import_path = SUPPORTED_BACKBONE_BACKENDS[backend]
    mm_backbones = importlib.import_module(backend_import_path)
    mm_registry = mm_backbones.BACKBONES
    custom_imports.append(backend_import_path)
    return mm_registry, custom_imports


def get_module_args(module):
    """Gather module's Required Args."""
    if module is None:
        return []
    required_args = []
    default_args = {}
    args_signature = inspect.signature(module)
    for arg_key, arg_value in args_signature.parameters.items():
        if arg_value.default is inspect.Parameter.empty:
            required_args.append(arg_key)
            continue
        default_args[arg_key] = arg_value.default
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
    return required_args, default_args


def get_otx_root_path():
    """Get otx root path from importing otx."""
    otx_module = importlib.import_module("otx")
    if otx_module:
        return os.path.dirname(inspect.getfile(otx_module))
    return None
