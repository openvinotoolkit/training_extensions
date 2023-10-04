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
from pathlib import Path
from typing import Callable, Dict, List, TypeVar, Union


def get_impl_class(impl_path: str) -> TypeVar:
    """Returns a class by its path in package."""

    task_impl_module = None
    result = None
    try:
        task_impl_module_name, task_impl_class_name = impl_path.rsplit(".", 1)
        task_impl_module = importlib.import_module(task_impl_module_name)
        result = getattr(task_impl_module, task_impl_class_name)

    except Exception as e:
        if hasattr(task_impl_module, "DEBUG"):
            exception = getattr(task_impl_module, "DEBUG", None)
            if isinstance(exception, Exception):
                raise exception from e
        raise e from None
    return result


def get_default_args(func: Callable) -> list:
    signature = inspect.signature(func)
    non_default_args = []

    for name, parameter in signature.parameters.items():
        if parameter.default is not inspect.Parameter.empty and parameter.default is not None:
            non_default_args.append((name, parameter.default))

    return non_default_args


def get_all_args(func: Callable) -> List[str]:
    signature = inspect.signature(func)
    return list(signature.parameters.keys())


def get_otx_root_path() -> str:
    """Get otx root path from importing otx."""
    otx_module = importlib.import_module("otx")
    if otx_module:
        file_path = inspect.getfile(otx_module)
        return str(Path(file_path).parent)
    msg = "Cannot found otx."
    raise ModuleNotFoundError(msg)


def get_files_dict(folder_path: Union[str, Path]) -> Dict[str, str]:
    file_path_dict = {}

    _folder_path: Path = Path(folder_path)
    if not _folder_path.exists():
        msg = "The specified folder path does not exist."
        raise ValueError(msg)

    for file_path in _folder_path.iterdir():
        if file_path.is_file():
            file_name = file_path.stem
            file_path_dict[file_name] = str(file_path)

    return file_path_dict
