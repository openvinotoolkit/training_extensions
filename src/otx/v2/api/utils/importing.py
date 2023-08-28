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
import os
from pathlib import Path
from typing import Dict


def get_impl_class(impl_path):
    """Returns a class by its path in package."""

    try:
        task_impl_module_name, task_impl_class_name = impl_path.rsplit(".", 1)
        task_impl_module = importlib.import_module(task_impl_module_name)
        task_impl_class = getattr(task_impl_module, task_impl_class_name)

        return task_impl_class
    except Exception:
        raise ImportError(task_impl_module_name)


def get_non_default_args(func):
    signature = inspect.signature(func)
    non_default_args = []

    for name, parameter in signature.parameters.items():
        if parameter.default is not inspect.Parameter.empty and parameter.default is not None:
            non_default_args.append((name, parameter.default))

    return non_default_args


def get_all_args(func):
    signature = inspect.signature(func)
    return signature.parameters.keys()


def get_otx_root_path():
    """Get otx root path from importing otx."""
    otx_module = importlib.import_module("otx")
    if otx_module:
        return os.path.dirname(inspect.getfile(otx_module))
    return None


def get_files_dict(folder_path) -> Dict[str, str]:
    file_path_dict = {}

    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise ValueError("The specified folder path does not exist.")

    for file_path in folder_path.iterdir():
        if file_path.is_file():
            file_name = file_path.stem
            file_path_dict[file_name] = str(file_path)

    return file_path_dict
